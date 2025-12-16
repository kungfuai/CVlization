import copy
import os
from typing import List, Optional

import torch
import torch._dynamo
import torch.distributed as dist

torch._dynamo.config.suppress_errors = True

import logging
import queue
import time
import uuid
from threading import Thread

from einops import rearrange

import self_forcing.utils.parallel_state as mpu
from config.config import config as service_config
from core import comm_utils
from core.distributed import broadcast_dict, recv_dict
from self_forcing.utils.misc import set_seed
from self_forcing.utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

logger = logging.getLogger(__name__)


def gpu_heartbeat(queue: queue.Queue):
    while True:
        logger.info(f"Putting gpu heartbeat signal: {queue.qsize()}")
        queue.put({"signal": "gpu_heartbeat"}, block=True)
        time.sleep(60)


class StreamCausalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device, profile=False):
        super().__init__()

        # Step 1: Initialize all models
        num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        independent_first_frame = getattr(args, "independent_first_frame", False)
        is_sparse_causal = getattr(args, "is_sparse_causal", False)
        local_attn_size = getattr(args, "local_attn_size", -1)
        if is_sparse_causal:
            sink_size = 1
        else:
            sink_size = 0
        causal_model_kwargs = {
            "num_frame_per_block": num_frame_per_block,
            "sink_size": sink_size,
            "independent_first_frame": independent_first_frame,
            "is_sparse_causal": is_sparse_causal,
            "local_attn_size": local_attn_size,
        }
        self.sink_size = sink_size
        self.is_sparse_causal = is_sparse_causal

        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}),
            is_causal=True,
            skip_init_model=True,
            **causal_model_kwargs,
        )

        self.text_encoder = WanTextEncoder()
        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long
        )
        if args.warp_denoising_step:
            timesteps = torch.cat(
                (self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))
            )
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = self.generator.model.num_layers
        self.num_heads = self.generator.model.num_heads
        self.dim = self.generator.model.dim
        self.text_seq_len = self.text_encoder.seq_len

        # self.latent_frame_size = args.image_or_video_shape[-2:]
        self.patch_size = self.generator.model.patch_size
        # self.frame_seq_length = self.latent_frame_size[0] * self.latent_frame_size[1] // self.patch_size[1] // self.patch_size[2]
        # self.frame_seq_length_sp = self.frame_seq_length // mpu.get_sequence_parallel_world_size()
        self.frame_seq_length = 0
        self.frame_seq_length_sp = 0

        self.kv_cache = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        self.local_attn_size = self.generator.model.local_attn_size
        self.device = device

        logger.info(f"KV inference with {self.num_frame_per_block} frames per block")

        self.profile = profile
        self.output_latent_queue = []

    @property
    def dtype(self):
        return self.generator.model.dtype

    def inference_init(self, conditional_dict: dict, sp_dim: Optional[str] = None):
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        if "ref_latents" in conditional_dict.keys():
            height, width = conditional_dict["ref_latents"].shape[-2:]
        else:
            raise NotImplementedError
        batch_size = 1

        if conditional_dict.get("motion_latents", None) is not None:
            self.sink_size = 2.5
        else:
            self.sink_size = 1

        # Set up profiling if requested
        if self.profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            cache_init_start = torch.cuda.Event(enable_timing=True)
            cache_init_end = torch.cuda.Event(enable_timing=True)
            init_start.record()
            cache_init_start.record()

        # Step 1: Initialize KV cache to all zeros
        old_frame_seq_length_sp = self.frame_seq_length_sp
        self.frame_seq_length_sp = (
            height * width // self.patch_size[1] // self.patch_size[2]
        )
        if self.kv_cache is None or old_frame_seq_length_sp != self.frame_seq_length_sp:
            self._initialize_kv_cache(
                batch_size=batch_size, dtype=self.dtype, device=self.device
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size, dtype=self.dtype, device=self.device
            )

        else:
            # reset cross attn cache
            self.reset_crossattn_cache()
            # reset kv cache
            for block_index in range(len(self.kv_cache)):
                self.kv_cache[block_index]["global_end_index"].fill_(0)
                self.kv_cache[block_index]["local_end_index"].fill_(0)
                # self.kv_cache[block_index]["global_end_index"] = 0
                # self.kv_cache[block_index]["local_end_index"] = 0

        self.current_start_frame = 0
        self.current_start_token = (
            0 if not service_config.lip_sync.no_refresh_inference else torch.tensor(0)
        )
        if self.profile:
            cache_init_end.record()
            torch.cuda.synchronize()
            cache_init_time = cache_init_start.elapsed_time(cache_init_end)
            logger.info(f"  - Clear caching time: {cache_init_time:.2f} ms")

        self.s2v_prefill(conditional_dict=conditional_dict, sp_dim=sp_dim)
        self.output_latent_queue = []

        if self.profile:
            init_end.record()
            torch.cuda.synchronize()
            init_time = init_start.elapsed_time(init_end)
            logger.info(f"  - Initialization/caching time: {init_time:.2f} ms")

    def s2v_prefill(self, conditional_dict, sp_dim: Optional[str] = None):
        if self.generator.model_type == "s2v":
            batch_size = 1
            timestep = torch.zeros(
                [batch_size, 1], device=self.device, dtype=torch.int64
            )
            ref_length = self.generator(
                noisy_image_or_video=conditional_dict["ref_latents"],
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=self.kv_cache,
                crossattn_cache=self.crossattn_cache,
                current_start=self.current_start_token,
                sp_dim=sp_dim,
                initial_ref=True,
                sink_size=self.sink_size,
                disable_float_conversion=True,
            )
            self.current_start_token += ref_length

    def inference_one_block(
        self,
        conditional_dict,
        batch_size=1,
        num_input_frames=0,
        cond_videos=None,
        face_mask=None,
        sp_dim: Optional[str] = None,
        audio_ptr: int = 0,
        **kwargs,
    ):
        if "ref_latents" in conditional_dict.keys():
            height, width = conditional_dict["ref_latents"].shape[-2:]
        else:
            raise NotImplementedError

        if self.profile:
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            block_start.record()

        noisy_input = torch.randn(
            [1, self.num_frame_per_block, 16, height, width],
            device=self.device,
            dtype=self.dtype,
        )
        current_num_frames = self.num_frame_per_block
        if conditional_dict.get("image_latent", None) is not None:
            block_conditional_dict = copy.copy(conditional_dict)
            if (
                self.current_start_frame + current_num_frames - num_input_frames
                < conditional_dict["image_latent"].shape[2]
            ):
                block_conditional_dict["image_latent"] = copy.copy(
                    conditional_dict["image_latent"][
                        :,
                        :,
                        self.current_start_frame
                        - num_input_frames : self.current_start_frame
                        + current_num_frames
                        - num_input_frames,
                    ]
                )
            else:
                block_conditional_dict["image_latent"] = copy.copy(
                    conditional_dict["image_latent"][:, :, -current_num_frames:]
                )
        else:
            block_conditional_dict = conditional_dict

        slice_index = [audio_ptr, audio_ptr + current_num_frames]

        # Step 3.1: Spatial denoising loop
        for index, current_timestep in enumerate(self.denoising_step_list):
            logger.info(f"current_timestep: {current_timestep}")
            # set current timestep
            timestep = (
                torch.ones(
                    [batch_size, current_num_frames],
                    device=self.device,
                    dtype=torch.int64,
                )
                * current_timestep
            )

            if index < len(self.denoising_step_list) - 1:
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=block_conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=self.current_start_token,
                    sp_dim=sp_dim,
                    slice_index=slice_index,
                    sink_size=self.sink_size,
                    disable_float_conversion=True,
                    **kwargs,
                )
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = (
                    self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones(
                            [batch_size * current_num_frames],
                            device=self.device,
                            dtype=torch.long,
                        ),
                    )
                    .unflatten(0, denoised_pred.shape[:2])
                    .to(self.dtype)
                )
            else:
                # for getting real output
                _, denoised_pred = self.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=block_conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=self.current_start_token,
                    sp_dim=sp_dim,
                    slice_index=slice_index,
                    sink_size=self.sink_size,
                    disable_float_conversion=True,
                    **kwargs,
                )

        # Step 3.2: record the model's output
        output = denoised_pred.to(self.dtype)

        # Step 3.3: rerun with timestep zero to update KV cache using clean context
        context_timestep = torch.ones_like(timestep) * self.args.context_noise
        self.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=block_conditional_dict,
            timestep=context_timestep,
            kv_cache=self.kv_cache,
            crossattn_cache=self.crossattn_cache,
            current_start=self.current_start_token,
            sp_dim=sp_dim,
            slice_index=slice_index,
            sink_size=self.sink_size,
            disable_float_conversion=True,
            **kwargs,
        )

        if self.profile:
            block_end.record()
            torch.cuda.synchronize()
            block_time = block_start.elapsed_time(block_end)
            logger.info(f"  - Block generation time: {block_time:.2f} ms")

        # Step 3.4: update the start and end frame indices
        self.current_start_frame += current_num_frames
        self.current_start_token += self.frame_seq_length_sp * current_num_frames

        return output

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache = []
        if self.is_sparse_causal:
            if self.generator.model_type == "s2v":
                if service_config.lip_sync.no_refresh_inference:
                    sink_size = self.sink_size
                else:
                    sink_size = 2.5
                kv_cache_size = round(
                    (sink_size + self.num_frame_per_block * 2)
                    * self.frame_seq_length_sp
                )
            else:
                kv_cache_size = round(
                    (self.sink_size + self.num_frame_per_block * 2)
                    * self.frame_seq_length_sp
                )
        else:
            if self.local_attn_size != -1:
                # Use the local attention size to compute the KV cache size
                kv_cache_size = self.local_attn_size * self.frame_seq_length_sp
            else:
                # Use the default KV cache size
                kv_cache_size = self.num_output_frames * self.frame_seq_length_sp

        for _ in range(self.num_transformer_blocks):
            kv_cache.append(
                {
                    "k": torch.empty(
                        [
                            batch_size,
                            kv_cache_size,
                            self.num_heads,
                            self.dim // self.num_heads,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.empty(
                        [
                            batch_size,
                            kv_cache_size,
                            self.num_heads,
                            self.dim // self.num_heads,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    # "global_end_index": 0,
                    # "local_end_index": 0
                }
            )

        self.kv_cache = kv_cache  # always store the clean cache

    def reset_crossattn_cache(self):
        """
        Reset cross-attention cache.
        """
        for block_index in range(self.num_transformer_blocks):
            self.crossattn_cache[block_index]["is_init"] = False

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append(
                {
                    "k": torch.empty(
                        [
                            batch_size,
                            self.text_seq_len,
                            self.num_heads,
                            self.dim // self.num_heads,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.empty(
                        [
                            batch_size,
                            self.text_seq_len,
                            self.num_heads,
                            self.dim // self.num_heads,
                        ],
                        dtype=dtype,
                        device=device,
                    ),
                    "is_init": False,
                }
            )
        self.crossattn_cache = crossattn_cache


def main():
    config = service_config.lip_sync.dit_config

    seed = getattr(service_config.lip_sync, "seed", 0)
    # Initialize distributed inference
    if "LOCAL_RANK" in os.environ:
        set_seed(seed + mpu.get_sequence_parallel_rank())
        device = torch.cuda.current_device()

    else:
        device = torch.device("cuda")
        set_seed(seed)

    torch.set_grad_enabled(False)

    logger.info("Rank %d Initializing pipeline" % mpu.get_rank())
    pipeline = StreamCausalInferencePipeline(
        config, device=device, profile=service_config.lip_sync.profile
    )

    logger.info("Rank %d Loading checkpoint" % mpu.get_rank())
    checkpoint_path = getattr(service_config.lip_sync, "checkpoint_path", None)
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        pipeline.generator.load_state_dict(state_dict["generator"])

    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    logger.info("Rank %d Pipeline init done" % mpu.get_rank())

    if mpu.get_sequence_parallel_rank() == 0:
        signal_queue = queue.Queue(256)  # For receiving control signals
        ready_signal_queue = queue.Queue(2)
        cond_queue = queue.Queue(2)

        socket_server_task = Thread(
            target=comm_utils.run_socket_server,
            kwargs={
                "queue": signal_queue,
                "port": service_config.server.diffusion_socket_port,
            },
        )
        ready_socket_server_task = Thread(
            target=comm_utils.run_socket_server,
            kwargs={
                "queue": ready_signal_queue,
                "port": service_config.server.diffusion_ready_socket_port,
            },
        )
        gpu_heartbeat_task = Thread(
            target=gpu_heartbeat, kwargs={"queue": signal_queue}
        )

        socket_server_task.start()
        ready_socket_server_task.start()
        gpu_heartbeat_task.start()

    server_state = 0
    # state 0: waiting
    # state 1: generating stream

    torch.distributed.barrier()
    logger.info("Rank %d initialization barrier passed" % mpu.get_rank())

    conditional_dict = {}
    sp_dim = None
    audio_ptr = 0
    current_audio_length = 0
    audio_finish_flag = True
    profile = service_config.lip_sync.profile
    send_socket = None
    ready_socket = None

    while True:
        # Receive signals
        if mpu.get_sequence_parallel_rank() == 0:
            cond_dict = {}
            start = time.time()
            while not cond_queue.full():
                if server_state == 0:
                    data_dict = signal_queue.get(block=True)
                else:
                    try:
                        data_dict = signal_queue.get(block=False)

                    except queue.Empty:
                        data_dict = None
                        break

                if data_dict is not None:
                    signal = data_dict["signal"]
                    if signal in {"start", "update"}:
                        ready_socket = comm_utils.socket_send(
                            data={"signal": "ready to recv"},
                            port=service_config.server.app_ready_socket_port,
                            client_socket=ready_socket,
                        )
                        logger.info(
                            f"Rank {mpu.get_rank()} trying to recv_batch with signal: {signal}"
                        )
                        cond_dict = recv_dict(src=0, profile=profile)
                        logger.info(f"Rank {mpu.get_rank()} data received")
                        cond_dict["signal"] = signal
                        cond_dict["time"] = time.time()
                        cond_queue.put(cond_dict)

                    elif signal in {"stop", "gpu_heartbeat"}:
                        cond_queue.put(
                            {
                                "signal": signal,
                                "time": time.time(),
                                "uuid": str(uuid.uuid4()),
                            }
                        )
                        logger.info(f"Rank {mpu.get_rank()}, {signal} received")

                    if server_state == 0:
                        break
                else:
                    break

            if cond_queue.full():
                logger.info(
                    f"Rank {mpu.get_rank()}, cond_queue full, stop receiving..."
                )

        torch.distributed.barrier(group=mpu.get_sequence_parallel_group())

        if server_state == 0:
            if mpu.get_sequence_parallel_rank() == 0:
                try:
                    conditional_dict_diff = cond_queue.get(block=False)
                except queue.Empty:
                    conditional_dict_diff = {"time": time.time(), "uuid": ""}
            else:
                conditional_dict_diff = {"time": time.time(), "uuid": ""}

            torch.distributed.barrier(group=mpu.get_sequence_parallel_group())
            broadcast_dict(conditional_dict_diff)
            torch.cuda.synchronize()

            if conditional_dict_diff.get("signal", None) == "start":
                conditional_dict.update(conditional_dict_diff)
                sp_dim = conditional_dict.get("sp_dim", None)
                if mpu.get_sequence_parallel_world_size() > 1:
                    if "ref_latents" in conditional_dict.keys():
                        conditional_dict["ref_latents"] = torch.chunk(
                            conditional_dict["ref_latents"],
                            mpu.get_sequence_parallel_world_size(),
                            dim=-2 if sp_dim == "h" else -1,
                        )[mpu.get_sequence_parallel_rank()]

                conditional_dict["motion_latents"] = None

                pipeline.inference_init(
                    conditional_dict=conditional_dict, sp_dim=sp_dim
                )
                server_state = 1
                logger.info(f"Rank {mpu.get_rank()} Server state 0 -> 1")

                if "prompt_embeds" in conditional_dict_diff.keys():
                    pipeline.reset_crossattn_cache()

                if "audio_input" in conditional_dict_diff.keys():
                    audio_ptr = 0
                    current_audio_length = (
                        conditional_dict_diff["audio_input"].shape[-1] // 4
                    )
                    audio_finish_flag = False
                    print(
                        "Rank %d:" % mpu.get_rank(),
                        "audio received with length",
                        current_audio_length,
                    )

        elif server_state == 1:
            if audio_finish_flag:  # audio finished
                start = time.time()
                while True:
                    if (
                        mpu.get_sequence_parallel_rank() == 0
                    ):  # TODO: wait until current audio finish
                        try:
                            conditional_dict_diff = cond_queue.get(block=False)
                        except queue.Empty:
                            conditional_dict_diff = {"time": time.time(), "uuid": ""}
                    else:
                        conditional_dict_diff = {"time": time.time(), "uuid": ""}
                    torch.distributed.barrier(group=mpu.get_sequence_parallel_group())
                    broadcast_dict(conditional_dict_diff)
                    torch.cuda.synchronize()
                    if conditional_dict_diff.get("signal", None) == "gpu_heartbeat":
                        continue
                    else:
                        break

                if conditional_dict_diff.get("signal", None) == "update":
                    logger.info(f"Rank {mpu.get_rank()} updating conditional_dict")
                    conditional_dict.update(conditional_dict_diff)
                    sp_dim = conditional_dict["sp_dim"]
                    if (
                        mpu.get_sequence_parallel_world_size() > 1
                        and "ref_latents" in conditional_dict_diff.keys()
                    ):
                        conditional_dict["ref_latents"] = torch.chunk(
                            conditional_dict["ref_latents"],
                            mpu.get_sequence_parallel_world_size(),
                            dim=-2 if sp_dim == "h" else -1,
                        )[mpu.get_sequence_parallel_rank()]
                        pipeline._initialize_kv_cache(
                            batch_size=1, dtype=pipeline.dtype, device=pipeline.device
                        )

                    if "prompt_embeds" in conditional_dict_diff.keys():
                        pipeline.reset_crossattn_cache()

                    if "audio_input" in conditional_dict_diff.keys():
                        audio_ptr = 0
                        current_audio_length = (
                            conditional_dict_diff["audio_input"].shape[-1] // 4
                        )
                        audio_finish_flag = conditional_dict_diff.get("silence", False)
                        logger.info(
                            f"Rank {mpu.get_rank()}: audio received with shape {conditional_dict_diff['audio_input'].shape}"
                        )

                elif conditional_dict_diff.get("signal", None) == "stop":
                    logger.info(f"Rank {mpu.get_rank()} received signal: stop")
                    server_state = 0
                    logger.info(f"Rank {mpu.get_rank()} Server state 1 -> 0")

                    if mpu.get_sequence_parallel_rank() == 0:
                        logger.info(f"Rank {mpu.get_rank()} Flushing cond_queue")
                        while not cond_queue.empty():
                            tmp = cond_queue.get()

                    conditional_dict = {}
                    sp_dim = None
                    audio_ptr = 0
                    current_audio_length = 0
                    audio_finish_flag = True
                    continue

            if (
                not service_config.lip_sync.no_refresh_inference
                and pipeline.current_start_frame
                >= service_config.lip_sync.s2v_video_refresh_interval
            ):
                conditional_dict["motion_latents"] = rearrange(
                    torch.cat(pipeline.output_latent_queue, dim=1)[
                        :, -conditional_dict["motion_frames"][1] :, ...
                    ],
                    "b t c h w -> b c t h w",
                )
                pipeline.inference_init(
                    conditional_dict=conditional_dict, sp_dim=sp_dim
                )

            # Inference one block
            output_block = pipeline.inference_one_block(
                conditional_dict=conditional_dict,
                sp_dim=sp_dim,
                audio_ptr=min(
                    audio_ptr, current_audio_length - pipeline.num_frame_per_block
                ),
            )

            if not service_config.lip_sync.no_refresh_inference:
                pipeline.output_latent_queue.append(output_block)

            if not audio_finish_flag:
                audio_ptr += pipeline.num_frame_per_block
                if audio_ptr >= current_audio_length:
                    logger.info(
                        "Audio ptr: %d/%d, audio finished"
                        % (audio_ptr, current_audio_length)
                    )
                    audio_finish_flag = True
                else:
                    logger.info(
                        "Audio ptr: %d/%d, audio playing"
                        % (audio_ptr, current_audio_length)
                    )

            if mpu.get_sequence_parallel_world_size() > 1:
                output_list = [
                    torch.empty_like(output_block)
                    for i in range(mpu.get_sequence_parallel_world_size())
                ]
                torch.distributed.all_gather(
                    output_list, output_block, group=mpu.get_sequence_parallel_group()
                )
                output_block = torch.cat(output_list, dim=-2 if sp_dim == "h" else -1)

            # Send output block
            if mpu.get_sequence_parallel_rank() == 0:
                send_socket = comm_utils.socket_send(
                    data={
                        "signal": "output",
                        "id": conditional_dict.get("id", ""),
                        "shape": list(output_block.shape),
                        "time": time.time(),
                    },
                    port=service_config.server.app_socket_port,
                    client_socket=send_socket,
                )
                logger.info(
                    f"Rank {mpu.get_rank()}, send signal sent, waiting for ready signal."
                )

                ready_signal = ready_signal_queue.get(block=True)
                torch.distributed.barrier(group=mpu.get_sequence_parallel_group())
                start = time.time()
                logger.info(f"Rank {mpu.get_rank()}, ready signal received, sending.")

                dist.send(output_block.to(torch.bfloat16), dst=0)
                torch.cuda.synchronize()
                logger.info(
                    "  - Rank %d, output block sent. Send time: %.3fms"
                    % (mpu.get_rank(), (time.time() - start) * 1000)
                )

            else:
                torch.distributed.barrier(group=mpu.get_sequence_parallel_group())

            torch.distributed.barrier(group=mpu.get_sequence_parallel_group())


if __name__ == "__main__":
    main()
