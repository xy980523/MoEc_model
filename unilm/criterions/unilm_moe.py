# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
from omegaconf import II

import torch
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from unilm.criterions.unilm import UniLmConfig, UniLmLoss
from unilm.modules import gshard
from unilm.modules import tmoe
from unilm.modules.gshard import MOELayer as gshard_layer
from unilm.modules.tmoe import MOELayer as tmoe_layer


@dataclass
class UniLmMoeConfig(UniLmConfig):
    moe_gate_loss_wt: float = field(
        default=1.0,
        metadata={
            "help": "Weight associated with MoE gate loss"
                    "in the weighted sum of gate loss and cross entropy loss"
        }
    )
    moe_gate_loss_combine_method: str = field(
        default="average",
        metadata={
            "help": "Method of combining the gate loss from each MoE layers"
                    "('sum', 'average')"
        }
    )
    moe_gate_loss_transform: str = field(
        default="none",
        metadata={
            "help": "Transformation to apply to the gate loss ('none', 'neg_log')"
        }
    )
    moe_loss_scale: str = field(
        default='sample_size',
        metadata={
            "help": "moe_loss = moe_loss_scale * mos_loss, ('sample_size', '1', '0.15') "
        }
    )


@register_criterion("unilm_moe", dataclass=UniLmMoeConfig)
class UniLmMoeLoss(UniLmLoss):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """
    moe_logging_keys = [
        "overflow_expert1",        # average % of overflowed tokens from 1st expert
        "overflow_expert2",        # average % of overflowed tokens from 2nd expert
        "entropy_gating",          # average entropy of the gating distribution
        "expert1_balance_top",     # average cumulative % of tokens processed by the most used 20% 1st experts
        "expert1_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 1st experts
        "unused_expert1_count",    # average number of 1st experts which process no tokens
        "expert2_balance_top",     # average cumulative % of tokens processed by the most used 20% 2nd experts
        "expert2_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 2nd experts
        "unused_expert2_count",    # average number of 2nd experts which process no tokens
        # "all_to_all_cpu_time_ms",  # CPU time spent in all to all calls in milliseconds
        # "all_to_all_cuda_time_ms", # CUDA ttime spent in all to all calls in milliseconds
        "expert_0_assigned_tokens",
        "expert_1_assigned_tokens",
        "expert_2_assigned_tokens",
        "expert_3_assigned_tokens",
    ]

    def mask_lm_loss(self, model, sample, reduce):
        masked_tokens = sample["targets"].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum()

        if self.tpu:
            masked_tokens = None  # always project all tokens on TPU
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
        else:
            masked_tokens = torch.where(
                masked_tokens.any(),
                masked_tokens,
                masked_tokens.new([True]),
            )
        features, extra = model(src_tokens=sample["src_tokens"])
        if masked_tokens is not None:
            features = features[masked_tokens, :]
        logits = model.output_layer(features)

        targets = sample["targets"]
        if masked_tokens is not None:
            targets = targets[targets.ne(self.padding_idx)]

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        # l_aux 
        gate_loss = 0.0
        gate_count = 0
        for l_aux in extra["l_aux"]:
            if l_aux is not None:
                gate_loss += l_aux
                gate_count += 1
        if self.cfg.moe_gate_loss_combine_method == "average":
            gate_loss = gate_loss / gate_count
        if self.cfg.moe_gate_loss_transform == "neg_log":
            gate_loss = - torch.log(gate_loss)
        
        mos_loss_scale = getattr(self.cfg, 'moe_loss_scale', 'sample_size')
        if mos_loss_scale == 'sample_size':
            gate_loss = sample_size * gate_loss
        elif mos_loss_scale == '1':
            gate_loss = gate_loss
        elif mos_loss_scale == '0.15': # mask_prob
            gate_loss = 0.15 * gate_loss

        mlm_loss = loss
        loss = mlm_loss + self.cfg.moe_gate_loss_wt * gate_loss

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "mlm_loss": mlm_loss.data,
            "moe_loss": gate_loss.data if type(gate_loss) != float else 0,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }

        moe_metadata = self.get_moe_metadata(model)
        logging_output.update(moe_metadata)

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        UniLmMoeLoss.reduce_moe_metrics(logging_outputs)

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
        )
        if "mlm_loss" in logging_outputs[0]:
            mlm_loss_sum = sum(log.get("mlm_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "mlm_loss", mlm_loss_sum / sample_size / math.log(2), sample_size, round=3
            )
        if "seq2seq_loss" in logging_outputs[0]:
            seq2seq_loss_sum = sum(log.get("seq2seq_loss", 0) for log in logging_outputs)
            metrics.log_scalar(
                "seq2seq_loss", seq2seq_loss_sum / sample_size / math.log(2), sample_size, round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
    def get_moe_metadata(self, model):
        moe_logging_output = {}
        for key in UniLmMoeLoss.moe_logging_keys:
            total_val = 0
            count = 0
            for _, module in model.named_modules():
                if isinstance(module, gshard_layer):
                    total_val += module.metadata[key] if key in module.metadata else 0
                    count += 1
                elif isinstance(module, tmoe_layer):
                    total_val += module.metadata[key] if key in module.metadata else 0
                    count += 1
                # elif isinstance(module, BaseLayer):
                #     total_val += module.metadata[key] if key in module.metadata else 0
                #     count += 1
            if count == 0:
                return {}
            moe_logging_output[key] = total_val / count
        moe_logging_output["batch_count"] = 1
        return moe_logging_output

    @staticmethod
    def reduce_moe_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        moe_loss_sum = sum(log.get("moe_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "moe_gate_loss", moe_loss_sum / sample_size, sample_size, round=8
        )
        batch_count = sum(log.get("batch_count", 0) for log in logging_outputs)
        for key in UniLmMoeLoss.moe_logging_keys:
            if batch_count == 0:
                continue
            val = sum(log.get(key, 0) for log in logging_outputs)
            metrics.log_scalar(
                key, val / batch_count, batch_count, round=3
            )