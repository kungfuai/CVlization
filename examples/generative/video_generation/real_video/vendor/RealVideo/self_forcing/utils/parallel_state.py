import torch

_SEQUENCE_PARALLEL_GROUP = None

def initialize_parallel_states() -> None:
    global _SEQUENCE_PARALLEL_GROUP
    rank = torch.distributed.get_rank()
    if rank == 0: # VAE
        sp_ranks = [0]
    else:
        sp_ranks = range(1, torch.distributed.get_world_size())
    sp_group = torch.distributed.new_group(sp_ranks, group_desc='sequence_parallel_group')
    if rank in sp_ranks:
        _SEQUENCE_PARALLEL_GROUP = sp_group

def get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0

def get_sequence_parallel_group(check_initialized=True):
    """Get the sequence parallel group the caller rank belongs to."""
    if check_initialized:
        assert _SEQUENCE_PARALLEL_GROUP is not None, "sequence parallel group is not initialized"
    return _SEQUENCE_PARALLEL_GROUP

def get_sequence_parallel_world_size(*args, **kwargs):
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())

def get_sequence_parallel_rank(*args, **kwargs):
    return torch.distributed.get_rank(group=get_sequence_parallel_group())

def get_sequence_parallel_src_rank():
    return torch.distributed.get_process_group_ranks(get_sequence_parallel_group())[0]

def sequence_parallel_is_initialized():
    return get_sequence_parallel_group() is not None

def destroy_parallel_groups():
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = None