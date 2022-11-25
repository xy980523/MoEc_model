import torch

from fairseq import checkpoint_utils
from fairseq.distributed import utils as distributed_utils


def _get_moe_layer(args):
    if args.transformer_moe_layers:
        return args.transformer_moe_layers.split(',')
    if args.insert_transformer_moe_layers:
        return args.insert_transformer_moe_layers.split(',')
    moe_freq = max(getattr(args, 'encoder_moe_freq', 0),
                   getattr(args, 'moe_freq', 0))
    moe_layers = []
    for i in range(args.encoder_layers):
        is_moe_layer = moe_freq != 0 and (i + 1) % moe_freq == 0
        if is_moe_layer:
            moe_layers.append(str(i))
    return moe_layers


def load_moe_ckpt(args, model, is_electra=False):
    num_workers = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    worker_id = 0 if not torch.distributed.is_initialized() else distributed_utils.get_data_parallel_rank()
    expert_path = args.fine_tune_stage_restore_model_path + \
        "-rank-" + str(worker_id) + ".pt"
    state = checkpoint_utils.load_checkpoint_to_cpu(expert_path)

    num_expert_per_worker = args.moe_expert_count // num_workers

    original_body_key = 'body'
    if is_electra:
        original_body_key = 'discriminator'

    for i in range(num_expert_per_worker):
        print("loading expert {} to worker {}.".format(
            num_expert_per_worker * worker_id + i, worker_id))
        expert_path = args.fine_tune_stage_restore_model_path + \
            "-rank-" + str(num_expert_per_worker * worker_id + i) + ".pt"
        expert_state = checkpoint_utils.load_checkpoint_to_cpu(expert_path)
        for key in expert_state["model"].keys():
            for moe_layer in _get_moe_layer(args):
                moe_layer = int(moe_layer)
                en_key = "{}.layers.{}.moe_layer.experts.0.".format(
                    original_body_key, moe_layer)  # by default, one expert per gpu
                if key.find(en_key) != -1:
                    prefix_len = len(en_key)
                    new_en_key = "{}.layers.{}.moe_layer.experts.".format(
                        original_body_key, moe_layer)
                    new_key = new_en_key + str(i) + '.' + key[prefix_len:]
                    state["model"][new_key] = expert_state["model"][key]

    model.load_state_dict(state["model"], strict=True, args=args)
