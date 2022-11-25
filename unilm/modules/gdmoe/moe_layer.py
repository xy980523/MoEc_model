import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda import Event as CudaEvent
from torch.nn import Module, ModuleList
from fairseq import distributed_utils
from unilm.distributed import utils as unilm_dist_utils
from fairseq.distributed import utils as distributed_utils
import random

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    has_tutel, fused_cumsum_sub_one = True, tutel_moe.fast_cumsum_sub_one
except ModuleNotFoundError:
    has_tutel, fused_cumsum_sub_one = False, lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class _All2All_with_split(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        if input_splits is not None and not isinstance(input_splits, list):
            input_splits = input_splits.tolist()
        if output_splits is not None and not isinstance(output_splits, list):
            output_splits = output_splits.tolist()

        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        if input_splits is not None and output_splits is not None:
            xs_list = list(xs.split(input_splits, dim=0))
            for i in range(len(input_splits)):
                if input_splits[i] == 0:
                    xs_list[i] = xs.new_empty(size=[1] + list(xs.size()[1:]))
            ys_list = list(ys.split(output_splits, dim=0))
            for i in range(len(output_splits)):
                if output_splits[i] == 0:
                    ys_list[i] = xs.new_empty(size=[1] + list(xs.size()[1:]))
            torch.distributed.all_to_all(ys_list, xs_list)
            for i in range(len(output_splits)):
                if output_splits[i] == 0:
                    ys_list[i] = xs.new_empty(size=[0] + list(xs.size()[1:]))
            ys = torch.cat(ys_list, dim=0)
        else:
            torch.distributed.all_to_all_single(ys, xs)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        if ctx.input_splits is not None and ctx.output_splits is not None:
            grad_output_list = list(grad_output.split(ctx.output_splits, dim=0))
            for i in range(len(ctx.output_splits)):
                if ctx.output_splits[i] == 0:
                    grad_output_list[i] = grad_output.new_empty(size=[1] + list(grad_output.size()[1:]))
            result_list = list(result.split(ctx.input_splits, dim=0))
            for i in range(len(ctx.input_splits)):
                if ctx.input_splits[i] == 0:
                    result_list[i] = grad_output.new_empty(size=[1] + list(grad_output.size()[1:]))
            torch.distributed.all_to_all(result_list, grad_output_list)
            for i in range(len(ctx.input_splits)):
                if ctx.input_splits[i] == 0:
                    result_list[i] = grad_output.new_empty(size=[0] + list(grad_output.size()[1:]))
            result = torch.cat(result_list, dim=0)
        else:
            torch.distributed.all_to_all_single(result, grad_output)
        return result, None, None


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList], args, group: Optional[Any] = None, all2all_group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.expert_group = group if group is not None else unilm_dist_utils.get_moe_group(args.moe_expert_count)
        self.all2all_group = all2all_group if all2all_group is not None else unilm_dist_utils.get_all2all_group(args.moe_expert_count)
        for p in experts.parameters():
            p.expert = True  # type: ignore
        self.world_size = distributed_utils.get_world_size(self.expert_group)
        self.all2all_size = distributed_utils.get_world_size(self.all2all_group)
        self.num_local_experts = len(self.experts)   # 1
        self.worker_id = distributed_utils.get_data_parallel_rank()   # 2
        self.num_workers = distributed_utils.get_data_parallel_world_size()  # 3

        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0
        # expert_level_dropout
        self.exp_level_drop = args.exp_level_drop
        self.expnum = int((1 - self.exp_level_drop) * self.num_local_experts)
        #self.survive_expert_indices = list(range(self.expnum))
        self.survive_expert_indices = random.sample(range(0,self.num_local_experts),self.expnum)    
        self.update_freq = args.dropout_interval
        self.num_updates = 0
        self.var_coef = args.var_coef
        self.group_num = args.group_num
        self.coef_type = args.coef_type

    def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:


        if self.num_updates % self.update_freq == 0:
            self.survive_expert_indices = random.sample(range(0,self.num_local_experts),self.expnum)
        self.survive_expert_indices.sort()
        #print("survive_expert_indices:")
        #print(self.survive_expert_indices)
        self.num_updates += 1

        #print('self.experts:',len(self.experts))  # 8
        #print('self.expnum:',self.expnum)   # 6
        #print('self.world_size:',self.world_size)  # 1
        #print('self.all2all_size:',self.all2all_size)  # 4
        #print('self.num_local_experts:',self.num_local_experts)  # 8
        #print('self.worker_id:',self.worker_id )   # 0,1,2,3
        #print('self.num_workers:',self.num_workers)  # 4


        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        if input_padding_mask is not None:
            assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)
        
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = input_padding_mask.reshape(-1) if input_padding_mask is not None else None

        is_shuffle = getattr(self.args, 'token_shuffle', False) \
            and reshaped_input.requires_grad \
            and not getattr(self.args, 'fine_tune_stage', False)

        if is_shuffle:
            reshaped_input, shuffle_sort, shuffle_input_splits, shuffle_output_splits = self._shuffle_tokens(reshaped_input)
        
        token_per_expert = reshaped_input.size(0)

        
        num_all_experts = self.survive_expert_indices * self.num_workers
        
        '''
        num_all_experts = []
        for num in range(self.num_workers):
            sur_exp = [i+len(self.experts)*num for i in self.survive_expert_indices]
            num_all_experts = num_all_experts+sur_exp
        print('self.survive_expert_indices:',self.survive_expert_indices)
        print('num_all_experts :',num_all_experts )
        '''
        
        l_aux, sort_by_expert, input_splits, output_splits, self.metadata = self.gate(reshaped_input, reshaped_input_padding_mask, num_all_experts, self.var_coef, self.group_num,self.coef_type)
        #l_aux, sort_by_expert, input_splits, output_splits, self.metadata = self.gate(reshaped_input, reshaped_input_padding_mask)
        # Swap these tokens for the right ones for our expert
        routed_features = _All2All_with_split.apply(reshaped_input[sort_by_expert], output_splits.sum(dim=-1), input_splits.sum(dim=-1))         
        # print('routed_features:',routed_features.shape)
        # Forward Expert Net
        all_routed_features = self._resort(routed_features, input_splits)
        # print("input_splits:",input_splits.shape)
        # print("self.all2all_size:",self.all2all_size)

        n_expert_assigned_tokens = input_splits.reshape(self.all2all_size, len(self.survive_expert_indices)).sum(dim=0)
        # print("n_expert_assigned_tokens: ",n_expert_assigned_tokens.shape)   # 6

        routed_features_list = list(all_routed_features.split(n_expert_assigned_tokens.tolist(), dim=0))   
        #print('routed_features_list:',len(routed_features_list))   # 6


        for i, routed_features in enumerate(routed_features_list):
            expert_id = self.num_local_experts * self.worker_id + self.survive_expert_indices[i]
            # print(expert_id)
            if routed_features.size(0) > 0:
                if getattr(self.args, 'fine_tune_stage', False):
                    # NOTE we do not set capacity during finetuning
                    alpha = self.gate.compute_gating_alpha(routed_features, expert_id)
                    routed_features_list[i] = alpha * self.experts[expert_id](routed_features) + (1 - alpha) * routed_features
                else:
                    capacity = int(token_per_expert * self.args.capacity_factor)
                    if routed_features.size(0) <= capacity:
                        alpha = self.gate.compute_gating_alpha(routed_features, expert_id)
                        routed_features_list[i] = alpha * self.experts[expert_id](routed_features) + (1 - alpha) * routed_features
                    else:
                        alpha = self.gate.compute_gating_alpha(routed_features[:capacity], expert_id)
                        routed_features1 = alpha * self.experts[expert_id](routed_features[:capacity]) + (1 - alpha) * routed_features[:capacity]
                        routed_features2 = routed_features[capacity:]
                        routed_features_list[i] = torch.cat((routed_features1, routed_features2), dim=0)


        all_routed_features = torch.cat(routed_features_list, dim=0)
        all_routed_features = self._resort(all_routed_features, input_splits.transpose(0, 1))

        # Return to original worker and ordering
        combined_output = _All2All_with_split.apply(all_routed_features, input_splits.sum(dim=-1), output_splits.sum(dim=-1))[self.inverse_sort(sort_by_expert)]
        
        # Return to original if shuffle
        if is_shuffle:
            combined_output = _All2All_with_split.apply(combined_output, shuffle_input_splits, shuffle_output_splits)[self.inverse_sort(shuffle_sort)]

        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[:reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input_shape[0], :, :]

        self.record_all_to_all_stats()

        return combined_output, l_aux

    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        output = _AllToAll.apply(self.all2all_group, input)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += (cpu_end - cpu_start)
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self):
        # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
        record_a2a_perf_stats = getattr(self.args, 'record_a2a_perf_stats', False)
        if record_a2a_perf_stats:
            torch.cuda.synchronize()
            self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
        # reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []

    def _shuffle_tokens(self, features):
        shuffle_sort = torch.randperm(features.size(0), device=features.device)
        shuffle_output_splits = [int(features.size(0) / self.all2all_size) for i in range(self.all2all_size)]
        shuffle_output_splits[-1] = features.size(0) - sum(shuffle_output_splits[:-1])
        shuffle_output_splits = torch.tensor(shuffle_output_splits, device=features.device).long()
        shuffle_input_splits = _All2All_with_split.apply(shuffle_output_splits)
        features = _All2All_with_split.apply(features[shuffle_sort], shuffle_output_splits, shuffle_input_splits)
        return features, shuffle_sort, shuffle_input_splits, shuffle_output_splits
    
    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def _resort(self, features, input_splits):
        features_list = []
        ptr = 0
        for i in range(input_splits.size(0)):
            for j in range(input_splits.size(1)):
                offset = int(input_splits[i][j])
                x = features[ptr:ptr + offset]
                ptr += offset

                if i == 0:
                    features_list.append([x])
                else:
                    features_list[j].append(x)

        for j in range(len(features_list)):
            features_list[j] = torch.cat(features_list[j], dim=0)

        return torch.cat(features_list, dim=0)

'''
import logging
import time
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
from torch import Tensor
from torch.cuda import Event as CudaEvent
from torch.nn import Module, ModuleList
from fairseq import distributed_utils
from unilm.distributed import utils as unilm_dist_utils
from fairseq.distributed import utils as distributed_utils
import random

if TYPE_CHECKING:
    Base = Module[Tensor]
else:
    Base = Module

try:
    # To enable Tutel MoE optimizations:
    #   python3 -m pip install --user --upgrade git+https://github.com/microsoft/tutel@v0.1.x
    from tutel import moe as tutel_moe

    has_tutel, fused_cumsum_sub_one = True, tutel_moe.fast_cumsum_sub_one
except ModuleNotFoundError:
    has_tutel, fused_cumsum_sub_one = False, lambda mask: torch.cumsum(mask, dim=0) - 1

logger = logging.getLogger(__name__)


# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

# Based on https://github.com/pytorch/pytorch/pull/40762
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Tensor) -> Tensor:  # type: ignore
        ctx.group = group
        input = input.contiguous()
        output = torch.empty_like(input)
        if torch.distributed.is_initialized():
            dist.all_to_all_single(output, input, group=group)
        else:
            assert group is None
            output = input
        return output

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor]:
        return (None, _AllToAll.apply(ctx.group, *grad_output))


# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class _All2All_with_split(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        if input_splits is not None and not isinstance(input_splits, list):
            input_splits = input_splits.tolist()
        if output_splits is not None and not isinstance(output_splits, list):
            output_splits = output_splits.tolist()

        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        if input_splits is not None and output_splits is not None:
            xs_list = list(xs.split(input_splits, dim=0))
            for i in range(len(input_splits)):
                if input_splits[i] == 0:
                    xs_list[i] = xs.new_empty(size=[1] + list(xs.size()[1:]))
            ys_list = list(ys.split(output_splits, dim=0))
            for i in range(len(output_splits)):
                if output_splits[i] == 0:
                    ys_list[i] = xs.new_empty(size=[1] + list(xs.size()[1:]))
            torch.distributed.all_to_all(ys_list, xs_list)
            for i in range(len(output_splits)):
                if output_splits[i] == 0:
                    ys_list[i] = xs.new_empty(size=[0] + list(xs.size()[1:]))
            ys = torch.cat(ys_list, dim=0)
        else:
            torch.distributed.all_to_all_single(ys, xs)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        if ctx.input_splits is not None and ctx.output_splits is not None:
            grad_output_list = list(grad_output.split(ctx.output_splits, dim=0))
            for i in range(len(ctx.output_splits)):
                if ctx.output_splits[i] == 0:
                    grad_output_list[i] = grad_output.new_empty(size=[1] + list(grad_output.size()[1:]))
            result_list = list(result.split(ctx.input_splits, dim=0))
            for i in range(len(ctx.input_splits)):
                if ctx.input_splits[i] == 0:
                    result_list[i] = grad_output.new_empty(size=[1] + list(grad_output.size()[1:]))
            torch.distributed.all_to_all(result_list, grad_output_list)
            for i in range(len(ctx.input_splits)):
                if ctx.input_splits[i] == 0:
                    result_list[i] = grad_output.new_empty(size=[0] + list(grad_output.size()[1:]))
            result = torch.cat(result_list, dim=0)
        else:
            torch.distributed.all_to_all_single(result, grad_output)
        return result, None, None



class _All2All_with_split_v2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        if input_splits is not None and not isinstance(input_splits, list):
            input_splits = input_splits.tolist()
        if output_splits is not None and not isinstance(output_splits, list):
            output_splits = output_splits.tolist()

        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        if input_splits is not None and output_splits is not None:
            xs_list = list(xs.split(input_splits, dim=0))
            ys_list = list(ys.split(output_splits, dim=0))
            torch.distributed.all_to_all(ys_list, xs_list)
            ys = torch.cat(ys_list, dim=0)
        else:
            torch.distributed.all_to_all_single(ys, xs)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        if ctx.input_splits is not None and ctx.output_splits is not None:
            grad_output_list = list(grad_output.split(ctx.output_splits, dim=0))
            result_list = list(result.split(ctx.input_splits, dim=0))
            torch.distributed.all_to_all(result_list, grad_output_list)
            result = torch.cat(result_list, dim=0)
        else:
            torch.distributed.all_to_all_single(result, grad_output)
        return result, None, None


class MOELayer(Base):
    """MOELayer module which implements MixtureOfExperts as described in Gshard_.
    ::

        gate = Top2Gate(model_dim, num_experts)
        moe = MOELayer(gate, expert)
        output = moe(input)
        l_aux = moe.l_aux

    .. Gshard_: https://arxiv.org/pdf/2006.16668.pdf

    Args:
        gate (torch.nn.Module):
            gate network
        expert (torch.nn.Module):
            expert network
    """

    def __init__(self, gate: Module, experts: Union[Module, ModuleList], args, group: Optional[Any] = None, all2all_group: Optional[Any] = None) -> None:
        super().__init__()
        self.gate = gate
        if type(experts) == ModuleList:
            self.experts = cast(ModuleList, experts)
        else:
            self.experts = ModuleList([experts])
        self.expert_group = group if group is not None else unilm_dist_utils.get_moe_group(args.moe_expert_count)
        self.all2all_group = all2all_group if all2all_group is not None else unilm_dist_utils.get_all2all_group(args.moe_expert_count)
        for p in experts.parameters():
            p.expert = True  # type: ignore
        self.world_size = distributed_utils.get_world_size(self.expert_group)
        self.all2all_size = distributed_utils.get_world_size(self.all2all_group)
        self.num_local_experts = len(self.experts)   # 1
        self.worker_id = distributed_utils.get_data_parallel_rank()   # 2
        self.num_workers = distributed_utils.get_data_parallel_world_size()  # 3

        self.args = args
        self.in_generation = False
        self.a2a_cuda_event_intervals = []
        self.a2a_cpu_time_ms = 0.0
        # expert_level_dropout
        self.exp_level_drop = args.exp_level_drop
        self.expnum = int((1 - self.exp_level_drop) * self.num_local_experts)
        #self.survive_expert_indices = list(range(self.expnum))
        self.survive_expert_indices = random.sample(range(0,self.num_local_experts),self.expnum)    
        self.update_freq = args.dropout_interval
        self.num_updates = 0
        self.var_coef = args.var_coef
        self.group_num = args.group_num
        self.coef_type = args.coef_type

    def forward(self, *input: Tensor, input_padding_mask=None, **kwargs: Any) -> Tensor:


        if self.num_updates % self.update_freq == 0:
            self.survive_expert_indices = random.sample(range(0,self.num_local_experts),self.expnum)
        self.survive_expert_indices.sort()
        #print("survive_expert_indices:")
        #print(self.survive_expert_indices)
        self.num_updates += 1

        #print('self.experts:',len(self.experts))  # 8
        #print('self.expnum:',self.expnum)   # 6
        #print('self.world_size:',self.world_size)  # 1
        #print('self.all2all_size:',self.all2all_size)  # 4
        #print('self.num_local_experts:',self.num_local_experts)  # 8
        #print('self.worker_id:',self.worker_id )   # 0,1,2,3
        #print('self.num_workers:',self.num_workers)  # 4


        assert len(input) == 1, "only single input Tensor supported"
        input = input[0]
        assert len(input.shape) == 3, "input Tensor must have dimensions: (s)equence, (t)oken, (m)odel"
        if input_padding_mask is not None:
            assert len(input_padding_mask.shape) == 2, "input Tensor must have dimensions: (s)equence, (t)oken"
            assert input_padding_mask.shape[0] == input.shape[0]
            assert input_padding_mask.shape[1] == input.shape[1]
        # assert input.shape[0] % len(self.experts) == 0, "num tokens must be order of number of local experts"

        # Implement Algorithm 2 from GShard paper.
        d_model = input.shape[2]
        # Pad to expected batch size
        input_shape = list(input.shape)
        
        # Reshape into S tokens by dropping sequence dimension.
        reshaped_input = input.reshape(-1, d_model)
        reshaped_input_shape = reshaped_input.shape
        reshaped_input_padding_mask = input_padding_mask.reshape(-1) if input_padding_mask is not None else None

        is_shuffle = getattr(self.args, 'token_shuffle', False) \
            and reshaped_input.requires_grad \
            and not getattr(self.args, 'fine_tune_stage', False)

        if is_shuffle:
            reshaped_input, shuffle_sort, shuffle_input_splits, shuffle_output_splits = self._shuffle_tokens(reshaped_input)
        
        token_per_expert = reshaped_input.size(0)

        
        num_all_experts = self.survive_expert_indices * self.num_workers
        
        
        #num_all_experts = []
        #for num in range(self.num_workers):
        #    sur_exp = [i+len(self.experts)*num for i in self.survive_expert_indices]
        #    num_all_experts = num_all_experts+sur_exp
        #print('self.survive_expert_indices:',self.survive_expert_indices)
        #print('num_all_experts :',num_all_experts )
        
        
        l_aux, sort_by_expert, input_splits, output_splits, self.metadata = self.gate(reshaped_input, reshaped_input_padding_mask, num_all_experts, self.var_coef, self.group_num,self.coef_type)
        #l_aux, sort_by_expert, input_splits, output_splits, self.metadata = self.gate(reshaped_input, reshaped_input_padding_mask)
        
        if self.args.use_all2all_v2:
            # append pseudo-tokens to the end of reshaped_input
            reshaped_input = torch.cat([reshaped_input, reshaped_input.new_zeros(self.args.moe_expert_count, d_model)], dim=0)
            routed_features = _All2All_with_split_v2.apply(reshaped_input[sort_by_expert], output_splits.sum(dim=-1), input_splits.sum(dim=-1))         
        else:
            routed_features = _All2All_with_split.apply(reshaped_input[sort_by_expert], output_splits.sum(dim=-1), input_splits.sum(dim=-1))         

        # print('routed_features:',routed_features.shape)
        # Forward Expert Net
        all_routed_features = self._resort(routed_features, input_splits)
        # print("input_splits:",input_splits.shape)
        # print("self.all2all_size:",self.all2all_size)

        n_expert_assigned_tokens = input_splits.reshape(self.all2all_size, len(self.survive_expert_indices)).sum(dim=0)
        # print("n_expert_assigned_tokens: ",n_expert_assigned_tokens.shape)   # 6

        routed_features_list = list(all_routed_features.split(n_expert_assigned_tokens.tolist(), dim=0))   
        #print('routed_features_list:',len(routed_features_list))   # 6


        for i, routed_features in enumerate(routed_features_list):
            expert_id = self.num_local_experts * self.worker_id + self.survive_expert_indices[i]
            # print(expert_id)
            if routed_features.size(0) > 0:
                if getattr(self.args, 'fine_tune_stage', False):
                    # NOTE we do not set capacity during finetuning
                    alpha = self.gate.compute_gating_alpha(routed_features, expert_id)
                    routed_features_list[i] = alpha * self.experts[expert_id](routed_features) + (1 - alpha) * routed_features
                else:
                    capacity = int(token_per_expert * self.args.capacity_factor)
                    if routed_features.size(0) <= capacity:
                        alpha = self.gate.compute_gating_alpha(routed_features, expert_id)
                        routed_features_list[i] = alpha * self.experts[expert_id](routed_features) + (1 - alpha) * routed_features
                    else:
                        alpha = self.gate.compute_gating_alpha(routed_features[:capacity], expert_id)
                        routed_features1 = alpha * self.experts[expert_id](routed_features[:capacity]) + (1 - alpha) * routed_features[:capacity]
                        routed_features2 = routed_features[capacity:]
                        routed_features_list[i] = torch.cat((routed_features1, routed_features2), dim=0)


        all_routed_features = torch.cat(routed_features_list, dim=0)
        all_routed_features = self._resort(all_routed_features, input_splits.transpose(0, 1))

        # Return to original worker and ordering
        if self.args.use_all2all_v2:
            combined_output = _All2All_with_split_v2.apply(all_routed_features, input_splits.sum(dim=-1), output_splits.sum(dim=-1))[self.inverse_sort(sort_by_expert)]
            # remove pseudo-tokens from the end of combined_output
            combined_output = combined_output[:-self.args.moe_expert_count]
        else:
            combined_output = _All2All_with_split.apply(all_routed_features, input_splits.sum(dim=-1), output_splits.sum(dim=-1))[self.inverse_sort(sort_by_expert)]

        # Return to original if shuffle
        if is_shuffle:
            combined_output = _All2All_with_split_v2.apply(combined_output, shuffle_input_splits, shuffle_output_splits)[self.inverse_sort(shuffle_sort)]

        # Remove padding here when --max-tokens is specified and not --batch-size or --max-sentences
        combined_output = combined_output[:reshaped_input_shape[0], :]
        combined_output = combined_output.reshape(input.shape)
        combined_output = combined_output[:input_shape[0], :, :]

        self.record_all_to_all_stats()

        return combined_output, l_aux

    def prepare_for_inference_(self):
        self.in_generation = True

    def all_to_all_wrapper(self, input: Tensor):
        dummy_a2a = getattr(self.args, 'dummy_a2a', False)
        if dummy_a2a:
            input = input.contiguous()
            output = input.detach().clone()
            return input
        # always record times, since it is not a lot of overhead
        # if we do not log it we simply clear it off in record_all_to_all_stats
        cuda_start = torch.cuda.Event(enable_timing=True)
        cuda_end = torch.cuda.Event(enable_timing=True)
        cpu_start = time.time() * 1000
        cuda_start.record()
        output = _AllToAll.apply(self.all2all_group, input)
        cuda_end.record()
        cpu_end = time.time() * 1000
        self.a2a_cpu_time_ms += (cpu_end - cpu_start)
        self.a2a_cuda_event_intervals.append((cuda_start, cuda_end))
        return output

    def record_all_to_all_stats(self):
        # controlled via an argument as we want to minimize any impact from torch.cuda.synchronize()
        record_a2a_perf_stats = getattr(self.args, 'record_a2a_perf_stats', False)
        if record_a2a_perf_stats:
            torch.cuda.synchronize()
            self.metadata["all_to_all_cpu_time_ms"] = self.a2a_cpu_time_ms
            a2a_cuda_time_ms = 0.0
            for ev_start, ev_end in self.a2a_cuda_event_intervals:
                a2a_cuda_time_ms += ev_start.elapsed_time(ev_end)
            self.metadata["all_to_all_cuda_time_ms"] = a2a_cuda_time_ms
        # reset stats
        self.a2a_cpu_time_ms = 0.0
        self.a2a_cuda_event_intervals = []

    def _shuffle_tokens(self, features):
        shuffle_sort = torch.randperm(features.size(0), device=features.device)
        shuffle_output_splits = [int(features.size(0) / self.all2all_size) for i in range(self.all2all_size)]
        shuffle_output_splits[-1] = features.size(0) - sum(shuffle_output_splits[:-1])
        shuffle_output_splits = torch.tensor(shuffle_output_splits, device=features.device).long()
        shuffle_input_splits = _All2All_with_split_v2.apply(shuffle_output_splits)
        features = _All2All_with_split_v2.apply(features[shuffle_sort], shuffle_output_splits, shuffle_input_splits)
        return features, shuffle_sort, shuffle_input_splits, shuffle_output_splits
    
    def inverse_sort(self, order):
        # Creates an index that undoes a sort: xs==xs[order][inverse_sort(order)]
        return torch.empty_like(order).scatter_(0, order, torch.arange(0, order.size(0), device=order.device))

    def _resort(self, features, input_splits):
        features_list = []
        ptr = 0
        for i in range(input_splits.size(0)):
            for j in range(input_splits.size(1)):
                offset = int(input_splits[i][j])
                x = features[ptr:ptr + offset]
                ptr += offset

                if i == 0:
                    features_list.append([x])
                else:
                    features_list[j].append(x)

        for j in range(len(features_list)):
            features_list[j] = torch.cat(features_list[j], dim=0)

        return torch.cat(features_list, dim=0)
'''
