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
from omegaconf import II
from unilm.models.squad import SQuADHead

DEFAULT_MAX_TARGET_POSITIONS = 1024

logger = logging.getLogger(__name__)

@dataclass
class UniLMModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    attention_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability for attention weights"}
    )
    activation_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability after activation in FFN."}
    )
    encoder_embed_dim: int = field(
        default=512, metadata={"help": "encoder embedding dimension"}
    )
    encoder_output_dim: int = field(
        default=512, metadata={"help": "encoder output dimension"}
    )
    encoder_input_dim: int = field(
        default=512, metadata={"help": "encoder input dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=2048, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_layers: int = field(default=6, metadata={"help": "num encoder layers"})
    encoder_attention_heads: int = field(
        default=8, metadata={"help": "num encoder attention heads"}
    )
    encoder_normalize_before: bool = field(
        default=False, metadata={"help": "apply layernorm before each encoder block"}
    )
    no_encoder_final_norm: bool = field(
        default=False,
        metadata={"help": "don't add an extra layernorm after the last encoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings (outside self attention)"
        },
    )
    share_encoder_input_output_embed: bool = field(
        default=False, metadata={"help": "share encoder input and output embeddings"}
    )
    encoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the encoder"},
    )
    layernorm_embedding: bool = field(
        default=False, metadata={"help": "add layernorm to embedding"}
    )
    no_scale_embedding: bool = field(
        default=False, metadata={"help": "if True, dont scale embeddings"}
    )
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    offload_activations: bool = field(
        default=False,
        metadata={"help": "move checkpointed activations to CPU after they are used."},
    )
    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for encoder"}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    # config for Fully Sharded Data Parallel (FSDP) training
    min_params_to_wrap: int = field(
        default=DEFAULT_MIN_PARAMS_TO_WRAP,
        metadata={
            "help": (
                "minimum number of params for a layer to be wrapped with FSDP() when "
                "training with --ddp-backend=fully_sharded. Smaller values will "
                "improve memory efficiency, but may make torch.distributed "
                "communication less efficient due to smaller input sizes. This option "
                "is set to 0 (i.e., always wrap) when --checkpoint-activations or "
                "--offload-activations are passed."
            )
        }
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max target positions"}
    )
    pooler_activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use for pooler layer"}
    )
    pooler_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the masked_lm pooler layers"}
    )
    # options from other parts of the config
    # add_bos_token: bool = II("task.add_bos_token")
    # tokens_per_sample: int = II("task.tokens_per_sample")
    tpu: bool = II("common.tpu")
    task_moe: bool = field(
        default=False, metadata={"help": "enable task moe"}
    )
    num_experts: int = field(
        default=2, metadata={"help": "number of experts for task moe"}
    )
    rel_pos_buckets: int = field(
        default=0, metadata={"help": ""}
    )
    max_rel_pos: int = field(
        default=0, metadata={"help": ""}
    )
    rescale_init: bool = field(
        default=False, metadata={"help": ""}
    )
    ffn_layernorm: bool = field(
        default=False, metadata={"help": ""}
    )
    
    # 
    sharded_save: bool = field(
        default=False, metadata={"help": "save model in each gpu"}
    )

@register_model("unilm", dataclass=UniLMModelConfig)
class UniLMModel(BaseFairseqModel):

    def __init__(self, args, body, lm_head):
        super().__init__()
        self.args = args
        self.body = body
        self.lm_head = lm_head
        self.classification_heads = nn.ModuleDict()

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

        body = UniLMBody(
            args, task.target_dictionary, embed_tokens
        )

        lm_head = cls.build_lm_head(
            args, args.encoder_embed_dim, len(task.dictionary), args.activation_fn, weight=embed_tokens.weight
        )
        return cls(args, body, lm_head)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens
    
    @classmethod
    def build_lm_head(cls, args, embed_dim, output_dim, activation_fn, weight):
        return LMHead(embed_dim, output_dim, activation_fn, weight)
    
    def output_layer(self, features):
        return self.lm_head(features)
    
    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = ClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )
    
    def register_question_answering_head(self, name, num_classes=None):
        self.classification_heads[name] = SQuADHead(
            self.args.encoder_embed_dim,
        )
    
    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            [] if not hasattr(self, 'classification_heads')
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.'):
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)
            inner_dim = state_dict[prefix + 'classification_heads.' + head_name + '.dense.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        'deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    logger.info('Overwriting ' + prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v
    
    def forward(self, src_tokens=None, tgt_tokens=None, incremental_state=None, classification_head_name=None, **kwargs):
        x, extra = self.body(src_tokens, tgt_tokens, incremental_state, return_all_hiddens=True)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)

        return x, extra

class UniLMBody(FairseqIncrementalDecoder):

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = cfg.encoder_layerdrop
        self.share_input_output_embed = cfg.share_encoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.encoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.encoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.encoder_learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        if self.encoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_encoder_layer(cfg)
                for _ in range(cfg.encoder_layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.encoder_normalize_before and not cfg.no_encoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        
        if cfg.rel_pos_buckets > 0 and cfg.max_rel_pos > 0:
            self.relative_position = RelativePositionBias(
                num_buckets=cfg.rel_pos_buckets, 
                max_distance=cfg.max_rel_pos, 
                n_heads=cfg.encoder_attention_heads,
            )
        else:
            self.relative_position = None
        
        self.apply(init_bert_params)
        if safe_getattr(cfg, 'rescale_init', False):
            self.rescale_fixup()
        

    def build_encoder_layer(self, cfg):
        if safe_getattr(cfg, "task_moe", False):
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
    
    def build_self_attn_mask(self, x, bidirectional=True):
        dim = x.size(1)
        _mask = torch.zeros([dim, dim])
        if not bidirectional:
            _mask = torch.triu(
                utils.fill_with_neg_inf(_mask), 1
            )
        _mask = _mask.to(x).float()
        return _mask
    
    def build_seq2seq_attn_mask(self, x, y):
        x_dim, y_dim = x.size(1), y.size(1)
        _mask = torch.zeros([x_dim+y_dim, x_dim+y_dim])
        _mask = torch.triu(
                utils.fill_with_neg_inf(_mask), 1
        )
        _mask[:, :x_dim] = 0
        _mask = _mask.to(x).float()
        return _mask
    
    def rescale_fixup(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        
        for layer_id in range(len(self.layers)):
            layer = self.layers[layer_id]
            rescale(layer.self_attn.out_proj.weight.data, layer_id + 1)
            if safe_getattr(self.cfg, "task_moe", False):
                for i in range(self.cfg.num_experts):
                    rescale(layer.fc2[i].weight.data, layer_id + 1)
            else:
                rescale(layer.fc2.weight.data, layer_id + 1)

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
        for idx, layer in enumerate(self.layers):
            x, layer_attn, _ = layer(
                x,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                src_tokens=src_tokens,
                tgt_tokens=tgt_tokens,
            )
            inner_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # if self.project_out_dim is not None:
        #     x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        return self.output_projection(features)

    def max_positions(self):
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

class UniLMTaskMoeLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.encoder_embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg
        )

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1, self.fc2 = nn.ModuleList([]), nn.ModuleList([])

        self.fc1.extend(
            [
                self.build_fc1(
                    self.embed_dim,
                    cfg.encoder_ffn_embed_dim
                )
                for _ in range(cfg.num_experts)
            ]
        )

        self.fc2.extend(
            [
                self.build_fc2(
                    cfg.encoder_ffn_embed_dim,
                    self.embed_dim
                )
                for _ in range(cfg.num_experts)
            ]
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
        if safe_getattr(cfg, "ffn_layernorm", False):
            self.ffn_layer_norm = LayerNorm(cfg.encoder_ffn_embed_dim, elementwise_affine=False)
        else:
            self.ffn_layer_norm = None

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

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
        src_tokens = None,
        tgt_tokens = None,
        **kwargs,
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

        if src_tokens is not None and tgt_tokens is not None:
            x1, x2 = x[:src_tokens.size(1)], x[src_tokens.size(1):]
            
            x1 = self.activation_fn(self.fc1[0](x1))
            x1 = self.activation_dropout_module(x1)
            if self.ffn_layer_norm is not None:
                x1 = self.ffn_layer_norm(x1)
            x1 = self.fc2[0](x1)

            x2 = self.activation_fn(self.fc1[1](x2))
            x2 = self.activation_dropout_module(x2)
            if self.ffn_layer_norm is not None:
                x2 = self.ffn_layer_norm(x2)
            x2 = self.fc2[1](x2)

            x = torch.cat([x1, x2], dim=0)
        elif src_tokens is not None:
            x = self.activation_fn(self.fc1[0](x))
            x = self.activation_dropout_module(x)
            if self.ffn_layer_norm is not None:
                x = self.ffn_layer_norm(x)
            x = self.fc2[0](x)
        else:
            x = self.activation_fn(self.fc1[1](x))
            x = self.activation_dropout_module(x)
            if self.ffn_layer_norm is not None:
                x = self.ffn_layer_norm(x)
            x = self.fc2[1](x)

        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        
        return x, attn, None

class UniLMLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.embed_dim = cfg.encoder_embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg
        )

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder_normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder_ffn_embed_dim
        )
        self.fc2 = self.build_fc2(
            cfg.encoder_ffn_embed_dim,
            self.embed_dim
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
        if safe_getattr(cfg, "ffn_layernorm", False):
            self.ffn_layer_norm = LayerNorm(cfg.encoder_ffn_embed_dim, elementwise_affine=False)
        else:
            self.ffn_layer_norm = None

    def build_fc1(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

    def build_fc2(self, input_dim, output_dim):
        return nn.Linear(input_dim, output_dim)

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

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layer_norm is not None:
            x = self.ffn_layer_norm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)

        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        
        return x, attn, None

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class LMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the masked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

class RelativePositionBias(nn.Module):
    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=12):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, step=None):
        """ Compute binned relative position bias """
        step = 0 if step is None else step
        context_position = torch.arange(step, step + qlen, dtype=torch.long,
                                        device=self.relative_attention_bias.weight.device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long,
                                       device=self.relative_attention_bias.weight.device)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(self, batch_size, qlen, klen, step=None):
        return self.compute_bias(qlen, klen, step).repeat(batch_size, 1, 1, 1).view(-1, qlen, klen)  # shape (batch * num_heads, qlen, klen)

@register_model_architecture("unilm", "unilm_base")
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

        
        
@register_model_architecture("unilm", "unilm_tiny")
def tiny_unilm_architecture(args):
    args.encoder_embed_dim = safe_getattr(args, "encoder_embed_dim", 384)
    args.encoder_ffn_embed_dim = safe_getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_layers = safe_getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = safe_getattr(args, "encoder_attention_heads", 6)

    # pre layer norm
    args.encoder_normalize_before = safe_getattr(args, 'encoder_normalize_before', True)
    args.no_encoder_final_norm = safe_getattr(args, "no_encoder_final_norm", False)
    args.layernorm_embedding = safe_getattr(args, "layernorm_embedding", False)

    base_unilm_architecture(args)
    
        
        
@register_model_architecture("unilm", "unilm_xlarge_24L")
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


@register_model_architecture("unilm", "unilm_xlarge_24L_post")
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


@register_model_architecture("unilm", "unilm_xlarge_48L")
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


@register_model_architecture("unilm", "unilm_xlarge_36L")
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
