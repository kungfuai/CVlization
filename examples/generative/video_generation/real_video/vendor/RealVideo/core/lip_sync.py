import asyncio
import datetime
import json
import logging
import math
import os
import time
import traceback
import uuid
from typing import List, Optional

import numpy as np
import starlette
import torch
import torchvision.transforms as TT
import websockets
from einops import rearrange
from PIL import Image

from config.config import config as service_config
from core import comm_utils
from core.distributed import send_dict
from core.utils import encode_image_async, encode_image_to_base64
from self_forcing.utils import parallel_state as mpu
from self_forcing.utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
from self_forcing.wan.modules.audio_encoder import AudioEncoder

logger = logging.getLogger(__name__)


async def send_cond_worker_async(
    cond_queue: asyncio.Queue,
    ready_signal_queue: asyncio.Queue,
    vae_idle_event: asyncio.Event,
    profile=False,
):
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    client_socket = None
    while True:
        try:
            logger.debug("Rank %d send worker: waiting for cond_queue" % mpu.get_rank())
            signal, conditional_dict = await cond_queue.get()
            logger.debug(
                "Rank %d send worker: cond fetched, waiting vae idle with signal: %s"
                % (mpu.get_rank(), signal)
            )

            await asyncio.sleep(0)
            if not conditional_dict.get("silence", False):
                await vae_idle_event.wait()
            conditional_dict["uuid"] = str(uuid.uuid4())
            logger.debug(
                "Rank %d send worker: vae idle passed, sending signal %s"
                % (mpu.get_rank(), signal)
            )

            client_socket = comm_utils.socket_send(
                data={"signal": signal},
                port=service_config.server.diffusion_socket_port,
                client_socket=client_socket,
            )
            if signal == "stop":
                continue
            logger.debug(
                "Rank %d: socket send done, waiting for target ready" % mpu.get_rank()
            )

            ready_signal = await ready_signal_queue.get()
            logger.debug(
                "Rank %d: ready signal received, sending conditional dict, uuid: %s"
                % (mpu.get_rank(), conditional_dict["uuid"])
            )

            send_dict(conditional_dict, dst=1, profile=profile)
            logger.debug(
                "Rank %d: conditional dict sent, uuid: %s"
                % (mpu.get_rank(), conditional_dict["uuid"])
            )
            await asyncio.sleep(0.01)

        except Exception as e:
            logger.exception(f"Exception in send_cond_worker: {e}")
            logger.exception(traceback.format_exc())


def remap_image(img: torch.Tensor):
    img = (
        (torch.clamp((img + 1) / 2, min=0, max=1) * 255).to(torch.uint8).flip(dims=[-1])
    )
    img = img.cpu().numpy().astype(np.uint8)
    return img


def nearest_multiple_of_64(n):
    lower_multiple = (n // 64) * 64
    upper_multiple = (n // 64 + 1) * 64
    if abs(n - lower_multiple) < abs(n - upper_multiple):
        return lower_multiple
    else:
        return upper_multiple


def get_closest_aspect_ratio(aspect_ratio):
    target_ratios = [16 / 9, 4 / 3, 1.0]
    distances = [abs(aspect_ratio - ratio) for ratio in target_ratios]
    closest_idx = distances.index(min(distances))
    return target_ratios[closest_idx]


def read_image(image_path, image_size=None, max_image_area=262144):
    image = Image.open(image_path).convert("RGB")
    img_W, img_H = image.size
    area = img_H * img_W
    if image_size is None:
        resize_ratio = math.sqrt(max_image_area / area)
        img_H = round(img_H * resize_ratio)
        img_W = round(img_W * resize_ratio)

    if img_H < img_W:
        target_ratio = get_closest_aspect_ratio(img_W / img_H)
        if image_size is None:
            target_H = int(nearest_multiple_of_64(img_H))
        else:
            target_H = image_size
        target_W = int(nearest_multiple_of_64(target_H * target_ratio))
        if img_W / img_H > target_ratio:
            resize_H = target_H
            resize_W = int(img_W / img_H * resize_H)
        else:
            resize_W = target_W
            resize_H = int(img_H / img_W * resize_W)
    else:
        target_ratio = get_closest_aspect_ratio(img_H / img_W)
        if image_size is None:
            target_W = int(nearest_multiple_of_64(img_W))
        else:
            target_W = image_size
        target_H = int(nearest_multiple_of_64(target_W * target_ratio))
        if img_H / img_W > target_ratio:
            resize_W = target_W
            resize_H = int(img_H / img_W * resize_W)
        else:
            resize_H = target_H
            resize_W = int(img_W / img_H * resize_H)

    chained_trainsforms = []
    chained_trainsforms.append(TT.Resize(size=[resize_H, resize_W], interpolation=3))
    chained_trainsforms.append(TT.CenterCrop(size=[target_H, target_W]))
    chained_trainsforms.append(TT.ToTensor())
    transform = TT.Compose(chained_trainsforms)
    image = transform(image).unsqueeze(0)  # chw
    image = image * 2.0 - 1.0

    if image.shape[-2] < image.shape[-1]:
        sp_dim = "h"
    else:
        sp_dim = "w"
    return image, sp_dim


class LipSyncManager:
    """
    VAE and image/audio encoder. Rank 0 of the inference service.
    The process of SelfForcingLipSync including:
        1. Handling user interaction. It receives user input and get audio response from a LLM+TTS or voice LLM.
        2. Sending Audio to AR DiT service.
        3. Receiving generated latent blocks from AR DiT service.
        4. Decoding latent blocks by local VAE. Sending decoded frames to frontend.
    """

    def __init__(self, vae_idle_event: asyncio.Event):
        self.fps = service_config.lip_sync.fps
        self.predefined_frames = []
        self.vae_idle_event = vae_idle_event

        self.device = torch.cuda.current_device()
        self.vae = WanVAEWrapper().to(dtype=torch.bfloat16, device=self.device)
        self.text_encoder = WanTextEncoder().to(
            dtype=torch.bfloat16, device=self.device
        )

        self.s2v_audio_encoder = AudioEncoder(device=self.device)

        self.signal_queue = asyncio.Queue(8)  # For receiving control signals
        self.ready_signal_queue = asyncio.Queue(2)  # For receiving ready signals
        self.cond_queue = asyncio.Queue(128)  # For sending conditional dict

        self.socket_server_task = None
        self.ready_socket_server_task = None
        self.sender_task = None
        self.vae_task = None

        self.do_decode_control_queue = asyncio.Queue(1)

        self.conditional_dict = {}
        self.audio_storage = {}  # For return audio_base64
        self.websocket = None

        self.silence_audio = (
            torch.randn(
                (1, service_config.lip_sync.audio_min_length * 1000),
                dtype=torch.float32,
            )
            * 1e-2
        )

        self.silence_prompt = service_config.video.silence_prompt
        self.speaking_prompt = service_config.video.speaking_prompt

        self.service_running = asyncio.Event()
        self.service_running.clear()

        self.frame_count = 0
        self.profile = service_config.lip_sync.profile

        self.ready_socket = None

        torch.distributed.barrier()
        logger.info("Rank %d initialization barrier passed" % mpu.get_rank())

    async def process_control_message(self, message):
        if message["type"] == "control":
            if message["text"] == "stop decode":
                while True:
                    try:
                        self.do_decode_control_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

            elif message["text"] == "do decode":
                try:
                    self.do_decode_control_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

        elif message["type"] == "image_config":
            logger.info("Image config received")
            image_path = message.get("image_path", "")

            cond = await self.update_condition(
                image_path=image_path,
                text=self.silence_prompt,
                audio=self.silence_audio,
                silence=True,
            )
            await self.start(cond)

    async def connect_websocket(self, websocket):
        self.websocket = websocket

        if self.socket_server_task is None:
            server_ready_event = asyncio.Event()
            self.socket_server_task = asyncio.create_task(
                comm_utils.run_socket_server_async(
                    self.signal_queue,
                    service_config.server.app_socket_port,
                    server_ready_event,
                )
            )
            await server_ready_event.wait()

        if self.ready_socket_server_task is None:
            server_ready_event = asyncio.Event()
            self.ready_socket_server_task = asyncio.create_task(
                comm_utils.run_socket_server_async(
                    self.ready_signal_queue,
                    service_config.server.app_ready_socket_port,
                    server_ready_event,
                )
            )
            await server_ready_event.wait()

        if self.sender_task is None:
            self.sender_task = asyncio.create_task(
                send_cond_worker_async(
                    cond_queue=self.cond_queue,
                    ready_signal_queue=self.ready_signal_queue,
                    vae_idle_event=self.vae_idle_event,
                    profile=self.profile,
                )
            )

        if self.vae_task is None:
            self.vae_task = asyncio.create_task(
                self.vae_decode(
                    signal_queue=self.signal_queue, vae_idle_event=self.vae_idle_event
                )
            )

        logger.info("Tasks init done.")

        try:
            self.do_decode_control_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

        self.vae_idle_event.set()

        logger.info("Websocket connected")

    async def disconnect_websocket(self):
        if self.websocket is not None:
            try:
                await self.websocket.close()

            except Exception as e:
                logger.exception(f"Exception in disconnect_websocket: {e}")
                logger.exception(traceback.format_exc())

            await self.stop()
            self.websocket = None
            try:
                self.do_decode_control_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        logger.info("Websocket disconnected")

    async def send_frame(self, frame_data, frame_idx):
        if self.websocket:
            try:
                data = {
                    "type": "audio_image",
                    "audio": frame_data.get("audio_base64", ""),
                    "image": frame_data.get("image_base64", ""),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "frame_index": frame_idx,
                    "audio_finish_frame": math.ceil(
                        frame_idx
                        + frame_data.get("audio_length", 0) * service_config.video.fps
                    ),
                    "total_frames": frame_idx + 1,
                }

                await self.websocket.send_text(json.dumps(data))
                logger.info(f"Frame {frame_idx} sent")

            except (
                websockets.exceptions.ConnectionClosedOK,
                websockets.exceptions.ConnectionClosed,
                starlette.websockets.WebSocketDisconnect,
            ) as e:
                logger.exception(f"Websocket closed in send_frame, {e}")
                await self.disconnect_websocket()

            except RuntimeError as e:
                logger.exception(f"Websocket closed in runtime error: {e}")
                await self.disconnect_websocket()

            except Exception as e:
                logger.exception(f"Failed to send frame: {e}")
                await self.disconnect_websocket()

        else:
            logger.info("Websocket not available.")

    async def update_condition(
        self, image_path=None, text=None, audio=None, id=None, silence=False
    ):
        conditional_dict = {}
        start_time = time.time()
        if text is None:
            conditional_dict.update({})
        else:
            if self.profile:
                text_enc_start = torch.cuda.Event(enable_timing=True)
                text_enc_end = torch.cuda.Event(enable_timing=True)
                text_enc_start.record()

            conditional_dict.update(self.text_encoder(text_prompts=[text]))
            conditional_dict["text"] = text

            await asyncio.sleep(0)
            if not silence:
                await self.vae_idle_event.wait()

            if self.profile:
                text_enc_end.record()
                torch.cuda.synchronize()
                text_enc_time = text_enc_start.elapsed_time(text_enc_end)
                logger.info(f"  - Text encode time: {text_enc_time:.2f} ms")

        if id is not None:
            conditional_dict["id"] = id

        await asyncio.sleep(0)
        if not silence:
            await self.vae_idle_event.wait()

        if image_path is not None:
            if self.profile:
                image_enc_start = torch.cuda.Event(enable_timing=True)
                image_enc_end = torch.cuda.Event(enable_timing=True)
                image_enc_start.record()

            image, sp_dim = read_image(image_path=image_path)
            original_image = rearrange(image.squeeze(0), "c h w -> h w c")
            logger.info(f"Image size:, {original_image.shape}")

            self.predefined_frames = [original_image]

            image = image.to(self.device).unsqueeze(2)  # bcthw
            image_latent = self.vae.encode_to_latent(image.to(torch.bfloat16)).permute(
                0, 2, 1, 3, 4
            )  # bcthw
            if self.profile:
                image_enc_end.record()
                torch.cuda.synchronize()
                image_enc_time = image_enc_start.elapsed_time(image_enc_end)
                logger.info(f"  - Image read & encode time: {image_enc_time:.2f} ms")

            conditional_dict["ref_latents"] = image_latent.to(torch.bfloat16)  # bcthw
            conditional_dict["sp_dim"] = sp_dim
            await asyncio.sleep(0)

        if audio is not None:
            if self.profile:
                audio_enc_start = torch.cuda.Event(enable_timing=True)
                audio_enc_end = torch.cuda.Event(enable_timing=True)
                audio_enc_start.record()

            await asyncio.sleep(0)
            if not silence:
                await self.vae_idle_event.wait()

            z = self.s2v_audio_encoder.extract_audio_feat(
                audio_input=audio.squeeze(0), return_all_layers=True
            )

            await asyncio.sleep(0)
            if not silence:
                await self.vae_idle_event.wait()

            audio_embed_bucket, num_repeat = (
                self.s2v_audio_encoder.get_audio_embed_bucket_fps(
                    z,
                    fps=service_config.video.fps,
                    batch_frames=(audio.shape[1] // 1000),
                    m=0,
                )
            )

            audio_embed_bucket = audio_embed_bucket.to(self.device).to(torch.bfloat16)
            audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
            if len(audio_embed_bucket.shape) == 3:
                audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
            elif len(audio_embed_bucket.shape) == 4:
                audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)

            if self.profile:
                audio_enc_end.record()
                torch.cuda.synchronize()
                audio_enc_time = audio_enc_start.elapsed_time(audio_enc_end)
                logger.info(f"  - Audio encode time: {audio_enc_time:.2f} ms")

            conditional_dict["audio_input"] = audio_embed_bucket
            conditional_dict["num_repeat"] = num_repeat
            conditional_dict["motion_frames"] = [73, 19]

        conditional_dict.update({"cond_finish_time": time.time()})
        conditional_dict["silence"] = silence

        return conditional_dict

    async def start(self, conditional_dict):
        try:
            self.frame_count = 1
            self.service_running.set()

            await self.send_frame(
                frame_data={
                    "image_base64": encode_image_to_base64(
                        remap_image(self.predefined_frames[0])
                    ),
                    "time": time.time(),
                },
                frame_idx=0,
            )
            await self.cond_queue.put(("start", conditional_dict))

            await asyncio.sleep(0)
        except Exception as e:
            logger.exception(f"Exception in start:, {e}")
            logger.exception(traceback.format_exc())

    async def stop(self):
        await self.cond_queue.put(("stop", {"cond_finish_time": time.time()}))
        self.service_running.clear()
        self.audio_storage = {}

    async def process_audio_chunk(
        self, audio_base64: Optional[str], decoded_audio: Optional[torch.tensor]
    ) -> List:
        if audio_base64 is None and decoded_audio is None:  # silence
            id = str(uuid.uuid4())
            print("Rank %d" % mpu.get_rank(), "sending silence audio")
            cond_diff = await self.update_condition(
                audio=self.silence_audio, id=id, text=self.silence_prompt, silence=True
            )
            self.audio_storage[id] = {
                "data": "",
                "time": time.time(),
                "length": self.silence_audio.shape[-1]
                / service_config.audio.sample_rate,
            }
            await self.cond_queue.put(("update", cond_diff))

        else:
            audio_data, sample_rate = decoded_audio, service_config.audio.sample_rate
            id = str(uuid.uuid4())
            self.audio_storage[id] = {
                "data": audio_base64,
                "time": time.time(),
                "length": audio_data.shape[-1] / service_config.audio.sample_rate,
            }

            audio_data_16k = audio_data
            target_length = (
                math.ceil(
                    (
                        audio_data_16k.shape[1]
                        - (1000 * service_config.lip_sync.audio_padding_rem)
                    )
                    / (1000 * service_config.lip_sync.audio_padding_div)
                )
                * 1000
                * service_config.lip_sync.audio_padding_div
                + 1000 * service_config.lip_sync.audio_padding_rem
            )
            pad_length = target_length - audio_data_16k.shape[1]

            await asyncio.sleep(0)
            await self.vae_idle_event.wait()
            audio_data_16k = torch.cat(
                [
                    audio_data_16k,
                    torch.zeros(
                        size=(1, pad_length),
                        dtype=audio_data_16k.dtype,
                        device=audio_data_16k.device,
                    ),
                ],
                dim=1,
            )

            cond_diff = await self.update_condition(
                audio=audio_data_16k, id=id, text=self.speaking_prompt, silence=False
            )

            await self.cond_queue.put(("update", cond_diff))

    async def vae_decode(
        self, signal_queue: asyncio.Queue, vae_idle_event: asyncio.Event
    ):
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        while True:
            try:
                logger.info(
                    "Rank %d VAE: waiting do decode control event" % mpu.get_rank()
                )

                if self.service_running.is_set():
                    await self.do_decode_control_queue.get()
                    logger.info(
                        "Rank %d VAE: do decode control event passed, waiting for signal_queue"
                        % mpu.get_rank()
                    )
                else:
                    logger.info(
                        "Rank %d VAE: service not running, skipping do decode control event"
                        % mpu.get_rank()
                    )

                data_dict = await signal_queue.get()

                vae_idle_event.clear()
                logger.info("Rank %d Setting vae lock" % mpu.get_rank())

                if data_dict["signal"] == "output":
                    id = data_dict["id"]
                    shape = data_dict["shape"]

                    logger.info(
                        "Rank %d: trying to receive output block" % mpu.get_rank()
                    )
                    output_block = torch.empty(
                        shape, dtype=torch.bfloat16, device=torch.cuda.current_device()
                    )

                    self.ready_socket = comm_utils.socket_send(
                        data={"signal": "ready"},
                        port=service_config.server.diffusion_ready_socket_port,
                        client_socket=self.ready_socket,
                    )
                    logger.info(
                        "Rank %d: ready to receive output block" % mpu.get_rank()
                    )

                    torch.distributed.recv(output_block, src=1)
                    logger.info(
                        f"Rank {mpu.get_rank()}: output block received, {output_block.shape}"
                    )

                    output_block_dict = {
                        "output_block": output_block,
                        "id": id,
                        "shape": shape,
                    }

                else:
                    logger.info(
                        "Rank %d Releasing vae lock in incorrect signal"
                        % mpu.get_rank()
                    )
                    vae_idle_event.set()
                    continue

                if not self.service_running.is_set():
                    logger.info(
                        "Rank %d flushing output block in vae_decode" % mpu.get_rank()
                    )
                    vae_idle_event.set()
                    continue

                if self.profile:
                    vae_start = torch.cuda.Event(enable_timing=True)
                    vae_end = torch.cuda.Event(enable_timing=True)
                    vae_start.record()

                output_block = output_block_dict["output_block"]
                id = output_block_dict.get("id", None)

                frame_block = self.vae.decode_to_pixel(
                    output_block, use_cache=True
                )  # btchw
                frame_block = rearrange(frame_block, "b t c h w -> b t h w c").squeeze(
                    0
                )
                await asyncio.sleep(0)

                logger.info(
                    f"Rank {mpu.get_rank()}: vae decoded frame_block: {frame_block.shape}"
                )
                frames = remap_image(frame_block)
                await asyncio.sleep(0)

                if self.profile:
                    vae_end.record()
                    torch.cuda.synchronize()
                    vae_time = vae_start.elapsed_time(vae_end)
                    logger.info(f"  - VAE decode time: {vae_time:.2f} ms")
                    base64_start = time.time()

                frame_dicts = [
                    {"image_base64": await encode_image_async(frame)}
                    for frame in frames
                ]
                if self.profile:
                    base64_time = (time.time() - base64_start) * 1000
                    logger.info(f"  - Base64 encode time: {base64_time:.3f} ms")
                await asyncio.sleep(0)

                if id in self.audio_storage.keys():
                    audio_data = self.audio_storage.get(id, dict())
                    audio_base64 = audio_data.get("data", "")
                    audio_length = audio_data.get("length", 5)
                    frame_dicts[0].update(
                        {"audio_base64": audio_base64, "audio_length": audio_length}
                    )
                    self.audio_storage.pop(id)

                for f in frame_dicts:
                    # send
                    await self.send_frame(frame_data=f, frame_idx=self.frame_count)
                    self.frame_count += 1

                logger.info("Rank %d Releasing vae lock" % mpu.get_rank())
                vae_idle_event.set()

            except Exception as e:
                logger.exception(
                    f"Rank {mpu.get_rank()}, Releasing vae lock in exception {e}"
                )
                vae_idle_event.set()
