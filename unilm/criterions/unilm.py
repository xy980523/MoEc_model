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


@dataclass
class UniLmConfig(FairseqDataclass):
    mlm_only: bool = field(
        default=False, metadata={"help": "use mlm objective only"}
    )
    seq2seq_only: bool = field(
        default=False, metadata={"help": "use seq2seq objective only"}
    )
    tpu: bool = II("common.tpu")


@register_criterion("unilm", dataclass=UniLmConfig)
class UniLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.cfg = cfg
        self.mask_idx = task.mask_idx

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

        features = model(src_tokens=sample["src_tokens"])[0]
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

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "mlm_loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output
    
    def seq2seq_loss(self, model, sample, reduce):
        features = model(src_tokens=sample["src_tokens"], tgt_tokens=sample["tgt_tokens"])[0]
        features = features[:, sample["src_tokens"].size(1):, :]
        logits = model.output_layer(features)
        targets = sample["targets"]
        sample_size = targets.ne(self.padding_idx).int().sum()

        loss = modules.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="sum",
            ignore_index=self.padding_idx,
        )

        logging_output = {
            "loss": loss if self.tpu else loss.data,
            "seq2seq_loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, sample_size, logging_outputs = 0, 0, {}

        if "mlm" in sample and not self.cfg.seq2seq_only:
            mlm_loss, mlm_sample_size, mlm_logging_output = self.mask_lm_loss(
                model, sample["mlm"], reduce=reduce
            )
            loss += mlm_loss
            sample_size = mlm_sample_size
            logging_outputs.update(mlm_logging_output)
        
        if "seq2seq" in sample and not self.cfg.mlm_only:
            seq2seq_loss, seq2seq_sample_loss, seq2seq_logging_output = self.seq2seq_loss(
                model, sample["seq2seq"], reduce=reduce
            )
            if "mlm" in sample:
                loss += seq2seq_loss * sample_size / seq2seq_sample_loss
                logging_outputs["loss"] += seq2seq_logging_output["loss"]
                logging_outputs["ntokens"] += seq2seq_logging_output["ntokens"]
                logging_outputs["nsentences"] += seq2seq_logging_output["nsentences"]
                logging_outputs["seq2seq_loss"] = seq2seq_logging_output["seq2seq_loss"]
            else:
                logging_outputs.update(seq2seq_logging_output)
                loss, sample_size = seq2seq_loss, seq2seq_sample_size
        
        return loss, sample_size, logging_outputs

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
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
