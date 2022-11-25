import math
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import BaseFairseqModel, FairseqIncrementalDecoder, register_model, register_model_architecture
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    MultiheadAttention,
)
from fairseq.models.transformer import (
    DEFAULT_MIN_PARAMS_TO_WRAP, Embedding
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from torch import Tensor
from fairseq.utils import safe_getattr, safe_hasattr
from fairseq.modules.quant_noise import quant_noise
from fairseq import distributed_utils as dist_utils, utils
from omegaconf import II
from unilm.models.squad import SQuADHead
from unilm.models.unilm import UniLMModelConfig, UniLMModel, UniLMBody, UniLMTaskMoeLayer, UniLMLayer
from unilm.modules.gshard import Top1Gate, Top2Gate, MOELayer
from unilm.modules.gshard.utils import load_moe_ckpt
from unilm.modules.fused_bias_gelu import fused_bias_gelu, has_fused_bias_gelu

DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)

@dataclass
class UniLMGshardModelConfig(UniLMModelConfig):
    # Mixture of Expert Layer arguments
    alternate_decoder_ffn_embed_dim: int = field(
        default=0,
        metadata={
            "help": "decoder FFN embed dim of alternate decoder layers"
        },
    )
    moe_freq: int = field(
        default=0,
        metadata={
            "help": "Frequency at which we insert MoE Transformer layers"
        },
    )
    moe_expert_count: int = field(
        default=0,
        metadata={
            "help": "Number of experts in each MoE Layer"
        }
    )
    moe_gating_use_fp32: bool = field(
        default=False,
        metadata={
            "help": "Use FP32 computations in MoE top2 gating function"
        }
    )
    moe_second_expert_policy: str = field(
        default='sampling',
        metadata={
            "help": "policy for second expert, options: all/sampling/random"
        }
    )
    moe_normalize_gate_prob_before_dropping: bool = field(
        default=False,
        metadata={
            "help": 'whether to normalize gate probs before or after dropping experts for capacity and randomization'
        }
    )
    moe_expert_ffn_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "MoE expert FFN dimension"
        }
    )
    moe_top1_expert: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Use top1 gate instead of top2"
        }
    )
    moe_eval_capacity_token_fraction: Optional[float] = field(
        default=0.25,
        metadata={
            "help": "Default: 0.25, Fraction of tokens as capacity during validation, if set to negative, use same as training. range: (0.0, 1.0]."
        }
    )
    moe_normalize_expert_grad: Optional[str] = field(
        default='world_size',
        metadata={
            "help": "Divide expert gradients by (1) 'world_size' (2) 'sqrt_world_size'"
        }
    )
    use_moe_pad_mask: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Don't route padding tokens to any expert",
        }
    )
    record_a2a_perf_stats: Optional[bool] = field(
        default=False, metadata={"help": "records all to all perf stats during distributed training"}
    )
    dummy_a2a: Optional[bool] = field(
        default=False, metadata={"help": "By passes all to all during distributed training by returning the input buffer as output"}
    )
    moe_batch_prioritized_routing: Optional[bool] = field(
        default=False, metadata={"help": "if true orders token by the gate prob before capacity dropping."}
    )
    use_stable_embedding: Optional[bool] = field(
        default=False,
        metadata={"help": 'Use bitsandbytes StableEmbeddingLayer which saves embedding state in fp32',
                  'argparse_alias': "--stable-emb"}
    )
    transformer_moe_layers: str = field(
        default="", 
        metadata={
            "help": "Convert moe_freq(Frequency at which we insert MoE Transformer layers) into transformer_moe_layers if it is null"
        },)
    
    # parameters for fine-tuning 
    fine_tune_stage: bool = field(default=False)
    fine_tune_stage_restore_model_path: str = field(default="")
    

@register_model("unilm_gshard", dataclass=UniLMGshardModelConfig)
class UniLMGshardModel(UniLMModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        args.max_target_positions = safe_getattr(
            args, "max_target_positions", DEFAULT_MAX_TARGET_POSITIONS
        )

        embed_tokens = cls.build_embedding(
            args, task.source_dictionary, args.encoder_input_dim
        )

        body = UniLMGshardBody(
            args, task.target_dictionary, embed_tokens
        )

        lm_head = cls.build_lm_head(
            args, args.encoder_embed_dim, len(task.dictionary), args.activation_fn, weight=embed_tokens.weight
        )
        model = cls(args, body, lm_head)
        if args.fine_tune_stage:
            load_moe_ckpt(args, model)
        return model


class UniLMGshardBody(UniLMBody):

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(cfg, dictionary, embed_tokens, output_projection)
        
        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        
        # build encoder layer
        moe_freq = max(getattr(cfg, 'encoder_moe_freq', 0), getattr(cfg, 'moe_freq', 0))
        for i in range(cfg.encoder_layers):
            if cfg.transformer_moe_layers:
                transformer_moe_layers = cfg.transformer_moe_layers.split(',')
                is_moe_layer = str(i) in transformer_moe_layers
            else:
                is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
            self.layers.append(self.build_encoder_layer(cfg, is_moe_layer=is_moe_layer))
            
        self.num_layers = len(self.layers)
        
        self.apply(init_bert_params)
        if safe_getattr(cfg, 'rescale_init', False):
            self.rescale_fixup()
        

    def build_encoder_layer(self, cfg, is_moe_layer=False):
        if is_moe_layer:
            layer = UniLMMoeLayer(cfg)
        elif safe_getattr(cfg, "task_moe", False):
            layer = UniLMTaskMoeLayer(cfg)
        else:
            layer = UniLMLayer(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer
    

    def forward(
        self,
        src_tokens,
        tgt_tokens,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        return_all_hiddens: bool = False,
    ):
        # embed positions
        positions = None
        if self.embed_positions is not None:
            if src_tokens is not None:
                src_positions = self.embed_positions(
                    src_tokens, incremental_state=incremental_state
                )
            if tgt_tokens is not None:
                tgt_positions = self.embed_positions(
                    tgt_tokens, incremental_state=incremental_state
                )

        tokens, self_attn_mask = None, None

        if src_tokens is not None and tgt_tokens is not None:
            tokens = torch.cat([src_tokens, tgt_tokens], dim=1)
            self_attn_mask = self.build_seq2seq_attn_mask(src_tokens, tgt_tokens)
            if self.embed_positions is not None:
                positions = torch.cat([src_positions, tgt_positions], dim=1)
        elif src_tokens is not None:
            tokens = src_tokens
            self_attn_mask = self.build_self_attn_mask(src_tokens, bidirectional=True)
            if self.embed_positions is not None:
                positions = src_positions
        else:
            tokens = tgt_tokens
            self_attn_mask = self.build_self_attn_mask(tgt_tokens, bidirectional=False)
            if self.embed_positions is not None:
                positions = tgt_positions

        if self.relative_position is not None:
            rel_pos_bias = self.relative_position(
                batch_size=tokens.size(0), 
                qlen=tokens.size(1),
                klen=tokens.size(1)
            )
            self_attn_mask = self_attn_mask.unsqueeze(0) + rel_pos_bias

        if incremental_state is not None:
            tokens = tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = tokens.eq(self.padding_idx)

        # encoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        l_aux = []
        for idx, layer in enumerate(self.layers):
            x, layer_attn, l_aux_i = layer(
                x,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
            )
            inner_states.append(x)
            if isinstance(layer, UniLMMoeLayer):
                l_aux.append(l_aux_i)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # if self.project_out_dim is not None:
        #     x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "l_aux": l_aux}

    def rescale_fixup(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        for layer_id in range(len(self.layers)):
            layer = self.layers[layer_id]
            rescale(layer.self_attn.out_proj.weight.data, layer_id + 1)
            if safe_getattr(self.cfg, "task_moe", False):
                for i in range(self.cfg.num_experts):
                    rescale(layer.fc2[i].weight.data, layer_id + 1)
            if hasattr(layer, 'moe_layer'):
                for expert in layer.moe_layer.experts:
                    rescale(expert.fc2.weight.data, layer_id + 1)
            else:
                rescale(layer.fc2.weight.data, layer_id + 1)


class UniLMMoeLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.encoder_embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg
        )

        ffn_dim = cfg.encoder_ffn_embed_dim
        if getattr(cfg, "alternate_ffn_embed_dim", 0.0) > 0:
            ffn_dim = getattr(cfg, "alternate_ffn_embed_dim", 0.0)

        if cfg.moe_top1_expert:
            gate = Top1Gate(
                self.embed_dim,
                cfg.moe_expert_count,
                use_fp32=cfg.moe_gating_use_fp32,
                moe_eval_capacity_token_fraction=getattr(cfg, "moe_eval_capacity_token_fraction", 0.25),
            )
        else:
            gate = Top2Gate(
                self.embed_dim,
                cfg.moe_expert_count,
                cfg.moe_gating_use_fp32,
                cfg.moe_second_expert_policy,
                cfg.moe_normalize_gate_prob_before_dropping,
                getattr(cfg, "moe_eval_capacity_token_fraction", 0.25),
                getattr(cfg, "moe_batch_prioritized_routing", False),
            )
        experts = make_experts(cfg, self.embed_dim, ffn_dim, self.dropout_module)
        self.moe_layer = MOELayer(gate, experts, cfg)
        
        self.normalize_before = cfg.encoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.final_layer_norm = LayerNorm(self.embed_dim)
        if safe_getattr(cfg, "ffn_layernorm", False):
            self.ffn_layer_norm = LayerNorm(cfg.encoder_ffn_embed_dim, elementwise_affine=False)
        else:
            self.ffn_layer_norm = None

    def build_self_attention(
        self, embed_dim, cfg
    ):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder_attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=False,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        # x - seq_len, batch_size, model_dim
        x = x.transpose(0, 1) # batch_size, seq_len, model_dim
        if getattr(self.cfg, "use_moe_pad_mask", False):
            x, l_aux = self.moe_layer(x, input_padding_mask=self_attn_padding_mask)
        else:
            x, l_aux = self.moe_layer(x)
        x = x.transpose(0, 1) # seq_len, batch_size, model_dim

        # x = self.activation_fn(self.fc1(x))
        # x = self.activation_dropout_module(x)
        # if self.ffn_layer_norm is not None:
        #     x = self.ffn_layer_norm(x)
        # x = self.fc2(x)
        # x = self.dropout_module(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        
        return x, attn, l_aux


def _linear(x, weight, bias=None):
    return F.linear(x, weight, bias)


def _ffn(
    x,
    fc1,
    activation_fn,
    activation_dropout_module,
    fc2,
    dropout_module,
):
    x_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    if has_fused_bias_gelu and activation_fn == gelu:
        x = _linear(x, fc1.weight)
        x = fused_bias_gelu(x, fc1.bias)
        x = activation_dropout_module(x)
        x = _linear(x, fc2.weight, fc2.bias)
    else:
        x = _linear(x, fc1.weight, fc1.bias)
        x = activation_fn(x)
        x = activation_dropout_module(x)
        x = _linear(x, fc2.weight, fc2.bias)
    x = x.view(x_shape)
    x = dropout_module(x)
    return x


class FeedForwardNetwork(nn.Module):
    """
        Feed Forward Network layer in the Transformer model
    """
    def __init__(self, args, embed_dim, ffn_dim, dropout_module=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.fc1 = self.build_fc1(
            self.embed_dim,
            ffn_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.dropout_module = FairseqDropout(
                args.dropout, module_name=self.__class__.__name__
            ) if not dropout_module else dropout_module

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def forward(self, x):
        return _ffn(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            activation_dropout_module=self.activation_dropout_module,
            fc2=self.fc2,
            dropout_module=self.dropout_module,
        )
        return x


def make_experts(args, embed_dim, expert_ffn_dim, dropout_module) -> nn.ModuleList:
    world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    expert_list = []
    ddp_rank = dist_utils.get_data_parallel_rank()
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert args.moe_expert_count % world_size == 0, f'{args.moe_expert_count}, {world_size}'
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with utils.set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    # less experts than gpus
    else:
        assert world_size % args.moe_expert_count == 0, f'{world_size}, {args.moe_expert_count}'
        # initialize each FFN with the same seed on different GPUs
        with utils.set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    experts = nn.ModuleList(expert_list)
    return experts


@register_model_architecture("unilm_gshard", "unilm_gshard_base")
def base_unilm_architecture(args):
    if safe_hasattr(args, "encoder_final_norm"):
        args.no_encoder_final_norm = not args.encoder_final_norm

    args.dropout = safe_getattr(args, "dropout", 0.1)
    args.attention_dropout = safe_getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = safe_getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = safe_getattr(args, "pooler_dropout", 0.0)

    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 12)
    args.encoder_learned_pos = safe_getattr(args, "encoder_learned_pos", True)
    args.activation_fn = safe_getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = safe_getattr(args, "pooler_activation_fn", "tanh")

    args.encoder_layerdrop = safe_getattr(args, "encoder_layerdrop", 0)
    args.encoder_layers_to_keep = safe_getattr(args, "encoder_layers_to_keep", None)

    # args.add_bos_token = safe_getattr(args, "add_bos_token", False)
    args.no_token_positional_embeddings = safe_getattr(
        args, "no_token_positional_embeddings", False
    )
    args.share_encoder_input_output_embed = safe_getattr(
        args, "share_encoder_input_output_embed", True
    )
    args.encoder_output_dim = safe_getattr(
        args, "encoder_output_dim", args.encoder_embed_dim
    )
    args.encoder_input_dim = safe_getattr(args, "encoder_input_dim", args.encoder_embed_dim)

    # Model training is not stable without this
    args.encoder_normalize_before = safe_getattr(args, 'encoder_normalize_before', False)
    args.no_encoder_final_norm = safe_getattr(args, "no_encoder_final_norm", True)

    args.no_scale_embedding = safe_getattr(args, "no_scale_embedding", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)
    args.checkpoint_activations = safe_getattr(args, "checkpoint_activations", False)
    args.offload_activations = safe_getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True


@register_model_architecture("unilm_gshard", "unilm_gshard_xlarge_24L")
def xlarge_24L_unilm_architecture(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 2048)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 8192)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 32)

    # pre layer norm
    args.encoder_normalize_before = safe_getattr(args, 'encoder_normalize_before', True)
    args.no_encoder_final_norm = safe_getattr(args, "no_encoder_final_norm", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)

    base_unilm_architecture(args)


@register_model_architecture("unilm_gshard", "unilm_gshard_xlarge_24L_post")
def xlarge_24L_unilm_architecture(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 2048)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 8192)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 32)

    # pre layer norm
    args.encoder_normalize_before = safe_getattr(args, 'encoder_normalize_before', False)
    args.no_encoder_final_norm = safe_getattr(args, "no_encoder_final_norm", True)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", True)

    base_unilm_architecture(args)


@register_model_architecture("unilm_gshard", "unilm_gshard_xlarge_48L")
def xlarge_24L_unilm_architecture(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 6144)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 48)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 24)

    # pre layer norm
    args.encoder_normalize_before = safe_getattr(args, 'encoder_normalize_before', True)
    args.no_encoder_final_norm = safe_getattr(args, "no_encoder_final_norm", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)

    base_unilm_architecture(args)


@register_model_architecture("unilm_gshard", "unilm_gshard_xlarge_36L")
def xlarge_24L_unilm_architecture(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 1536)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 6144)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 36)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 24)

    # pre layer norm
    args.encoder_normalize_before = safe_getattr(args, 'encoder_normalize_before', True)
    args.no_encoder_final_norm = safe_getattr(args, "no_encoder_final_norm", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)

    base_unilm_architecture(args)
