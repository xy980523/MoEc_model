# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# Implementation of Top2Gating described in https://arxiv.org/pdf/2006.16668.pdf
# Code is inspired by Top2GatingOnLogits from lingvo:
#   https://github.com/tensorflow/lingvo/blob/21b8106c5f1d30a196c98eedc441d4fd70833b11/lingvo/core/moe_layers.py#L477

# NOTE: This is a mirror of the code in
# https://github.com/facebookresearch/fairscale/tree/master/fairscale/nn/moe

from typing import Callable, Dict, Tuple, Optional

import math
import torch
from torch import Tensor
import torch.nn.functional as F

from fairseq.distributed import utils as distributed_utils
import torch.distributed as dist
from .moe_layer import has_tutel, fused_cumsum_sub_one, _All2All_with_split
from .top2gate import one_hot, entropy


#use a fixed temperature to compute balance loss
TEMPERATURE_FOR_L_UAX=0.07

# logging
SAMPLE_FRACTION = 0.2
me_sum = 0
tokens_num = 0

# Assigns each token to the top k experts
def greedy_assignment(scores, k=1, num_experts=None):
    num_workers = distributed_utils.get_data_parallel_world_size()
    if num_experts is None:
        num_experts = num_workers

    token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
    token_to_workers, sort_ordering = torch.sort(token_to_workers)
    worker2token = sort_ordering // k

    # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
    output_splits = torch.zeros((num_experts,), dtype=torch.long, device=scores.device)
    workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
    output_splits[workers] = counts
    # Tell other workers how many tokens to expect from us
    num_experts_per_worker = num_experts // num_workers   
    # print("output_splits:",output_splits.shape) #24
    # print("num_workers:",num_workers)  # 4
    # print("num_experts_per_worker:",num_experts_per_worker)   #6
    output_splits = output_splits.reshape(num_workers, num_experts_per_worker)
    input_splits = _All2All_with_split.apply(output_splits)
    return worker2token, input_splits, output_splits


def top1gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    gating_temperature=None,
    var_coef=0.0,
    group_num=1,
    coef_type=0,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top2Gating on logits."""
    metadata = {}

    if use_fp32:
        logits = logits.float()

    gates = logits
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # print('num_tokens:', num_tokens)  # 7612
    # print('num_experts:', num_experts)  # 24

    with torch.no_grad():
        gates1_s = torch.sigmoid(gates / gating_temperature)


    sort_by_expert, input_splits, output_splits = greedy_assignment(gates1_s, num_experts=num_experts)

    # Compute l_aux
    # the switch-transformer-style balance loss
    all2all_size = distributed_utils.get_data_parallel_world_size()
    n_assigned_tokens = input_splits.reshape(all2all_size, -1).sum(dim=0) # [num_local_experts]
    n_assigned_tokens_gathered = [torch.zeros_like(n_assigned_tokens) for _ in range(all2all_size)]
    dist.all_gather(n_assigned_tokens_gathered, n_assigned_tokens)
    n_assigned_tokens_gathered = torch.stack(n_assigned_tokens_gathered) # [ #expert]

    temperature = TEMPERATURE_FOR_L_UAX #use a fixed temperature to compute balance loss
    gates = F.softmax(gates / temperature, dim=1)  # [token, #expert]
    me = torch.sum(gates, dim=0)  # [expert]
    
    # ! smooth to avoid stucking
    
    ce = n_assigned_tokens_gathered / torch.sum(n_assigned_tokens_gathered) + 1e-6 # [#expert]

    #print('me:', me)  #[808.1815, 769.5322, 325.2863, 808.1815, 769.5322, 325.2863, 808.1815,769.5322, 325.2863, 808.1815, 769.5322, 325.2863]
    #print('ce:',ce)   # [4.0216e-01, 4.3428e-01, 1.6356e-01],[1.0000e-06, 1.0000e-06, 1.0000e-06] ^^^
    if coef_type == 0 or len(me)%group_num !=0:
        var = torch.var(me)
    
    elif coef_type == 1: 
        me_reshape = me.view(group_num,-1) 
        var_me = torch.var(me_reshape, dim=1)
        var = torch.mean(var_me)
        #print('me_reshape:',me_reshape.shape)  
        #print('var_me:',var_me.shape)

    elif coef_type == 2:
        me_reshape = me.view(group_num,-1)    # [group, group_size] 
        var_me = torch.var(me_reshape, dim=1)   # [group]
        var = torch.mean(var_me)
        #print('var:',var)
        me_mean = torch.mean(gates, dim=0)  #  [expert]
        me_mean, me_indices = torch.sort(me_mean, descending=True)
        me_mean2 = torch.mean(me_mean.view(group_num,-1),dim=1)
        dif = me_mean2[0] - me_mean2[1]
        var = var/ torch.exp(dif)

    else:
        var = 0

    ce = ce.view(me.size())
    
    '''
    global me_sum
    global tokens_num

    me_sum += me
    tokens_num += num_tokens 
    me_avg = me_sum / tokens_num
    print('me_avg',me_avg)
    '''
    l_aux = torch.matmul(me, ce) + var_coef * var
    l_aux = l_aux * num_experts

    return l_aux, sort_by_expert, input_splits, output_splits, metadata


class Top1Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        cfg,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        input_noise_type=None,
        capacity_factor=1.0,
        moe_eval_capacity_token_fraction=None,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()

        self.cfg = cfg
        if cfg.tmoe_routing_dim_reduction:
            self.wg_reduction = torch.nn.Linear(model_dim, cfg.tmoe_routing_dim, bias=False)
            self.wg = torch.nn.Linear(cfg.tmoe_routing_dim, num_experts, bias=False)
        else:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        
        torch.nn.init.orthogonal_(self.wg.weight, gain=0.1)
        self.num_experts = num_experts

        if cfg.tmoe_routing_hard_cosine:
            # register gating parameters 
            # gating_score(x,y) = (w1*norm + b1)*cos_score + b2
            # norm = ||x||*||y||, cos_score = x^Ty/(||x||*||y||)
            gating_t = torch.FloatTensor(1).fill_(0.07)
            self.register_parameter("gating_t", torch.nn.Parameter(gating_t))

        self.use_fp32 = use_fp32
        self.input_noise_type = input_noise_type

    def forward(self, 
                    input: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, 
                    survive_expert_indices=[0],
                    var_coef=0.0,
                    group_num=1,
                    coef_type=0,
                    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:  # type: ignore
        # import pudb;pu.db;
        if self.cfg.tmoe_routing_dim_reduction:
            input = self.wg_reduction(input)
        if self.cfg.tmoe_routing_hard_cosine:
            #print('survive_expert_indices:', survive_expert_indices)   
            #print('self.wg.weight:',self.wg.weight)    #[32,16]
            wg_weight_after_drop = self.wg.weight[survive_expert_indices]
            # print('wg_weight_after_drop:',wg_weight_after_drop.shape)   [24,16]
            logits = self._cosine(input, wg_weight_after_drop)
        else:
            wg_weight_after_drop = self.wg.weight[survive_expert_indices]
            logits = input.matmul(wg_weight_after_drop.t()) 
            # logits = self.wg(input)
        logits = self._make_finite(logits)
        return top1gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            gating_temperature=self._get_gating_temperature(),
            var_coef=var_coef,
            group_num=group_num,
            coef_type=coef_type,
        )

    def compute_gating_alpha(self, input, expert_id):
        if self.cfg.tmoe_routing_dim_reduction:
            input = self.wg_reduction(input)
        expert_cent = self.wg.weight[expert_id]
        if self.use_fp32:
            orig_dtype = input.dtype
            input = input.float()
            expert_cent = expert_cent.float()
        logits = self._cosine(input, expert_cent[None, :])
        logits = self._make_finite(logits)
        alpha = torch.sigmoid(logits / self._get_gating_temperature())
        if self.use_fp32:
            return alpha.to(orig_dtype)
        return alpha
       
    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2, p=2.0, dim=1, eps=eps)
        return mat1.matmul(mat2.transpose(0, 1))

    def _get_gating_temperature(self, eps=1e-4):
        if not hasattr(self, "gating_t"):
            return 1
        if self.gating_t.data.item() < eps:
            return eps
        return self.gating_t
    
    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores

'''
from typing import Callable, Dict, Tuple, Optional

import math
import torch
from torch import Tensor
import torch.nn.functional as F

from fairseq.distributed import utils as distributed_utils
import torch.distributed as dist
from .moe_layer import has_tutel, fused_cumsum_sub_one, _All2All_with_split, _All2All_with_split_v2
from .top2gate import one_hot, entropy


#use a fixed temperature to compute balance loss
TEMPERATURE_FOR_L_UAX=0.07

# logging
SAMPLE_FRACTION = 0.2


# Assigns each token to the top k experts
def greedy_assignment(scores, k=1, num_experts=None, use_all2all_v2=False):
    num_workers = distributed_utils.get_data_parallel_world_size()
    if num_experts is None:
        num_experts = num_workers

    token_to_workers = torch.topk(scores, dim=1, k=k, largest=True).indices.view(-1)
    
    if use_all2all_v2:
        token_to_workers = torch.cat([token_to_workers, torch.arange(0, num_experts).to(token_to_workers.device).long()], dim=0)
        
    token_to_workers, sort_ordering = torch.sort(token_to_workers)
    worker2token = sort_ordering // k

    # Find how many tokens we're sending to each other worker (being careful for sending 0 tokens to some workers)
    output_splits = torch.zeros((num_experts,), dtype=torch.long, device=scores.device)
    workers, counts = torch.unique_consecutive(token_to_workers, return_counts=True)
    output_splits[workers] = counts
    # Tell other workers how many tokens to expect from us
    num_experts_per_worker = num_experts // num_workers   
    # print("output_splits:",output_splits.shape) #24
    # print("num_workers:",num_workers)  # 4
    # print("num_experts_per_worker:",num_experts_per_worker)   #6
    output_splits = output_splits.reshape(num_workers, num_experts_per_worker)
    input_splits = _All2All_with_split_v2.apply(output_splits)
    return worker2token, input_splits, output_splits


def top1gating(
    logits: torch.Tensor,
    input_mask: Optional[torch.Tensor] = None,
    use_fp32=False,
    gating_temperature=None,
    var_coef=0.0,
    group_num=1,
    coef_type=0,
    use_all2all_v2=False,
) -> Tuple[Tensor, Tensor, Tensor, Dict]:
    """Implements Top2Gating on logits."""
    metadata = {}

    if use_fp32:
        logits = logits.float()

    gates = logits
    metadata["entropy_gating"] = entropy(probs=gates).mean().detach()

    # gates has shape of SE
    num_tokens = gates.shape[0]
    num_experts = gates.shape[1]
    # print('num_tokens:', num_tokens)  # 7612
    # print('num_experts:', num_experts)  # 24

    with torch.no_grad():
        gates1_s = torch.sigmoid(gates / gating_temperature)


    sort_by_expert, input_splits, output_splits = greedy_assignment(gates1_s, num_experts=num_experts, use_all2all_v2=use_all2all_v2)

    # Compute l_aux
    # the switch-transformer-style balance loss
    all2all_size = distributed_utils.get_data_parallel_world_size()
    n_assigned_tokens = input_splits.reshape(all2all_size, -1).sum(dim=0) # [num_local_experts]

    if use_all2all_v2:
        n_assigned_tokens = n_assigned_tokens - num_experts # for all2all_v2, pseudo-tokens are not counted

    n_assigned_tokens_gathered = [torch.zeros_like(n_assigned_tokens) for _ in range(all2all_size)]
    dist.all_gather(n_assigned_tokens_gathered, n_assigned_tokens)
    n_assigned_tokens_gathered = torch.stack(n_assigned_tokens_gathered) # [ #expert]

    temperature = TEMPERATURE_FOR_L_UAX #use a fixed temperature to compute balance loss
    gates = F.softmax(gates / temperature, dim=1)  # [token, #expert]
    me = torch.sum(gates, dim=0)  # [expert]
    
    # ! smooth to avoid stucking
    
    ce = n_assigned_tokens_gathered / torch.sum(n_assigned_tokens_gathered) + 1e-6 # [#expert]

    #print('me:', me)  #[808.1815, 769.5322, 325.2863, 808.1815, 769.5322, 325.2863, 808.1815,769.5322, 325.2863, 808.1815, 769.5322, 325.2863]
    #print('ce:',ce)   # [4.0216e-01, 4.3428e-01, 1.6356e-01],[1.0000e-06, 1.0000e-06, 1.0000e-06] ^^^
    #if coef_type == 0:
    if coef_type == 0 or len(me)%group_num !=0:
        var = torch.var(me)
    
    elif coef_type == 1:
        me_reshape = me.view(group_num,-1) 
        var_me = torch.var(me_reshape, dim=1)
        var = torch.mean(var_me)
        #print('me_reshape:',me_reshape.shape)  
        #print('var_me:',var_me.shape)

    elif coef_type == 2:
        me_reshape = me.view(group_num,-1)    # [group, group_size] 
        var_me = torch.var(me_reshape, dim=1)   # [group]
        var = torch.mean(var_me)
        #print('var:',var)
        me_mean = torch.mean(gates, dim=0)  #  [expert]
        me_mean, me_indices = torch.sort(me_mean, descending=True)
        me_mean2 = torch.mean(me_mean.view(group_num,-1),dim=1)
        dif = me_mean2[0] - me_mean2[1]
        var = var/ torch.exp(dif)

    else:
        var = 0

    ce = ce.view(me.size())
    l_aux = torch.matmul(me, ce) + var_coef * var
    l_aux = l_aux * num_experts

    return l_aux, sort_by_expert, input_splits, output_splits, metadata


class Top1Gate(torch.nn.Module):
    """Gate module which implements Top2Gating as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        l_aux, combine_weights, dispatch_mask = gate(input)

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        model_dim (int):
            size of model embedding dimension
        num_experts (ints):
            number of experts in model
    """

    wg: torch.nn.Linear

    def __init__(
        self,
        cfg,
        model_dim: int,
        num_experts: int,
        use_fp32=False,
        input_noise_type=None,
        capacity_factor=1.0,
        moe_eval_capacity_token_fraction=None,
    ) -> None:
        # TODO: merge this to top2gate.py
        #
        super().__init__()

        self.cfg = cfg
        if cfg.tmoe_routing_dim_reduction:
            self.wg_reduction = torch.nn.Linear(model_dim, cfg.tmoe_routing_dim, bias=False)
            self.wg = torch.nn.Linear(cfg.tmoe_routing_dim, num_experts, bias=False)
        else:
            self.wg = torch.nn.Linear(model_dim, num_experts, bias=False)
        
        torch.nn.init.orthogonal_(self.wg.weight, gain=0.1)
        self.num_experts = num_experts

        if cfg.tmoe_routing_hard_cosine:
            # register gating parameters 
            # gating_score(x,y) = (w1*norm + b1)*cos_score + b2
            # norm = ||x||*||y||, cos_score = x^Ty/(||x||*||y||)
            gating_t = torch.FloatTensor(1).fill_(0.07)
            self.register_parameter("gating_t", torch.nn.Parameter(gating_t))

        self.use_fp32 = use_fp32
        self.input_noise_type = input_noise_type

    def forward(self, 
                    input: torch.Tensor, 
                    mask: Optional[torch.Tensor] = None, 
                    survive_expert_indices=[0],
                    var_coef=0.0,
                    group_num=1,
                    coef_type=0,
                    ) -> Tuple[Tensor, Tensor, Tensor, Dict]:  # type: ignore
        # import pudb;pu.db;
        if self.cfg.tmoe_routing_dim_reduction:
            input = self.wg_reduction(input)
        if self.cfg.tmoe_routing_hard_cosine:
            #print('survive_expert_indices:', survive_expert_indices)   
            #print('self.wg.weight:',self.wg.weight)    #[32,16]
            wg_weight_after_drop = self.wg.weight[survive_expert_indices]
            # print('wg_weight_after_drop:',wg_weight_after_drop.shape)   [24,16]
            logits = self._cosine(input, wg_weight_after_drop)
        else:
            wg_weight_after_drop = self.wg.weight[survive_expert_indices]
            logits = input.matmul(wg_weight_after_drop.t()) 
            # logits = self.wg(input)
        logits = self._make_finite(logits)
        return top1gating(
            logits,
            mask,
            use_fp32=self.use_fp32,
            gating_temperature=self._get_gating_temperature(),
            var_coef=var_coef,
            group_num=group_num,
            coef_type=coef_type,
            use_all2all_v2=self.cfg.use_all2all_v2,
        )

    def compute_gating_alpha(self, input, expert_id):
        if self.cfg.tmoe_routing_dim_reduction:
            input = self.wg_reduction(input)
        expert_cent = self.wg.weight[expert_id]
        if self.use_fp32:
            orig_dtype = input.dtype
            input = input.float()
            expert_cent = expert_cent.float()
        logits = self._cosine(input, expert_cent[None, :])
        logits = self._make_finite(logits)
        alpha = torch.sigmoid(logits / self._get_gating_temperature())
        if self.use_fp32:
            return alpha.to(orig_dtype)
        return alpha
       
    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2, p=2.0, dim=1, eps=eps)
        return mat1.matmul(mat2.transpose(0, 1))

    def _get_gating_temperature(self, eps=1e-4):
        if not hasattr(self, "gating_t"):
            return 1
        if self.gating_t.data.item() < eps:
            return eps
        return self.gating_t
    
    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
'''
