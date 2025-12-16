import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist

from self_forcing.utils import parallel_state as mpu

logger = logging.getLogger(__name__)


def launch_distributed_job(backend: str = "nccl"):
    print("backend:", backend)
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    host = os.environ["MASTER_ADDR"]
    port = int(os.environ["MASTER_PORT"])

    if ":" in host:  # IPv6
        init_method = f"tcp://[{host}]:{port}"
    else:  # IPv4
        init_method = f"tcp://{host}:{port}"
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend=backend,
        init_method=init_method,
        timeout=timedelta(minutes=60),
    )
    mpu.initialize_parallel_states()
    torch.cuda.set_device(local_rank)


def send_dict(data, dst=None, profile=False):
    if profile:
        send_start = torch.cuda.Event(enable_timing=True)
        send_end = torch.cuda.Event(enable_timing=True)
        send_start.record()

    num_keys = [len(data.keys())]
    torch.distributed.send_object_list(num_keys, dst=dst)

    key_shape_obj_list = [
        (
            key,
            (data[key].shape, data[key].dtype)
            if isinstance(data[key], torch.Tensor)
            else data[key],
        )
        for key in data.keys()
    ]  # torch.Shape for Tensor, obj for others
    torch.distributed.send_object_list(key_shape_obj_list, dst=dst)

    for key_shape_obj in key_shape_obj_list:
        key, shape_or_obj = key_shape_obj
        if (
            isinstance(shape_or_obj, tuple)
            and isinstance(shape_or_obj[0], torch.Size)
            and isinstance(shape_or_obj[1], torch.dtype)
        ):  # torch.Tensor
            data[key] = data[key].to("cuda").contiguous()
            torch.distributed.send(data[key], dst=dst)

        else:
            data[key] = shape_or_obj

    if profile:
        send_end.record()
        torch.cuda.synchronize()
        send_time = send_start.elapsed_time(send_end)
        logger.info(f"  - Rank {mpu.get_rank()}, Send time: {send_time:.2f} ms")

    return data


def recv_dict(data=None, src=None, profile=False):
    if profile:
        recv_start = torch.cuda.Event(enable_timing=True)
        recv_end = torch.cuda.Event(enable_timing=True)
        recv_start.record()

    if data is None:
        data = {}
    num_keys = [None]
    torch.distributed.recv_object_list(num_keys, src=src)

    key_shape_obj_list = [None] * num_keys[0]  # torch.Shape for Tensor, obj for others
    torch.distributed.recv_object_list(key_shape_obj_list, src=src)

    for key_shape_obj in key_shape_obj_list:
        key, shape_or_obj = key_shape_obj
        if (
            isinstance(shape_or_obj, tuple)
            and isinstance(shape_or_obj[0], torch.Size)
            and isinstance(shape_or_obj[1], torch.dtype)
        ):  # torch.Tensor
            data[key] = torch.empty(
                shape_or_obj[0],
                device=torch.cuda.current_device(),
                dtype=shape_or_obj[1],
            )
            torch.distributed.recv(data[key], src=src)

        else:
            data[key] = shape_or_obj

    if profile:
        recv_end.record()
        torch.cuda.synchronize()
        recv_time = recv_start.elapsed_time(recv_end)
        logger.info(f"  - Rank {mpu.get_rank()}, Recv time: {recv_time:.2f} ms")

    return data


def broadcast_dict(data):
    group = mpu.get_sequence_parallel_group()
    src = mpu.get_sequence_parallel_src_rank()
    rank = mpu.get_sequence_parallel_rank()

    if rank == 0:
        num_keys = [len(data.keys())]
    else:
        num_keys = [None]
    torch.distributed.broadcast_object_list(num_keys, src=src, group=group)
    if num_keys[0] == 0:
        return {}

    if rank == 0:
        key_shape_obj_list = [
            (
                key,
                (data[key].shape, data[key].dtype)
                if isinstance(data[key], torch.Tensor)
                else data[key],
            )
            for key in data.keys()
        ]  # torch.Shape for Tensor, obj for others
    else:
        key_shape_obj_list = [None] * num_keys[0]
    torch.distributed.broadcast_object_list(key_shape_obj_list, src=src, group=group)

    for key_shape_obj in key_shape_obj_list:
        key, shape_or_obj = key_shape_obj
        if (
            isinstance(shape_or_obj, tuple)
            and isinstance(shape_or_obj[0], torch.Size)
            and isinstance(shape_or_obj[1], torch.dtype)
        ):  # torch.Tensor
            if rank != 0:
                data[key] = torch.empty(
                    shape_or_obj[0], device="cuda", dtype=shape_or_obj[1]
                )
            else:
                data[key] = data[key].to("cuda").contiguous()
            torch.distributed.broadcast(data[key], src=src, group=group)

        else:  # object
            if rank != 0:
                data[key] = shape_or_obj

    return data
