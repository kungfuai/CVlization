import asyncio
import base64
import logging
import time
import traceback
from typing import Any, Optional

import torch
import torchaudio

from config.config import config as service_config
from core.lip_sync import LipSyncManager
from core.tts_pipeline import TTSPipeline
from core.utils import encode_audio_to_base64

logger = logging.getLogger(__name__)


class ModelHandler:
    def __init__(self):
        self.vae_idle_event = asyncio.Event()
        self.vae_idle_event.set()

        self.lip_sync_manager = LipSyncManager(vae_idle_event=self.vae_idle_event)
        self.tts_pipeline = TTSPipeline(vae_idle_event=self.vae_idle_event)

        self.audio_count = 0

        self.audio_chunk_length = 1000 * service_config.lip_sync.audio_segment_length
        self.text_input_queue = asyncio.Queue(16)
        self.audio_chunk_queue = asyncio.Queue(64)  # audio output chunk

        self.audio_process_task = None
        self.audio_watermark_length = 0.6
        self.websocket = None

        self.audio_chunk_sizes = [1, 1, 3, 5]

    async def start_jobs(self, websocket):
        self.websocket = websocket
        self.tts_pipeline.start_async_tasks(
            text_input_queue=self.text_input_queue, output_queue=self.audio_chunk_queue
        )
        if self.audio_process_task is None:
            ready_event = asyncio.Event()
            ready_event.clear()
            self.audio_process_task = asyncio.create_task(
                self.process_audio(ready_event)
            )
            await ready_event.wait()
            logger.info("Audio processing task started")

    async def process_message(
        self,
        audio_base64: Optional[str] = None,
        sample_rate: Optional[int] = None,
        profile_content: Optional[str] = None,
        text_content: Optional[str] = None,
        voice_id: Optional[str] = None,
        websocket: Optional[Any] = None,
    ) -> None:
        if audio_base64 and len(audio_base64) < 100:
            logger.warning("Possibly invalid audio")
            audio_base64 = None
        chunk_count = 0
        self.websocket = websocket

        try:
            if text_content is not None:
                await self.text_input_queue.put(
                    {
                        "profile": profile_content,
                        "text": text_content,
                        "voice_id": voice_id,
                    }
                )
            elif audio_base64 is not None:
                await self.audio_chunk_queue.put(
                    {
                        "audio_base64": audio_base64,
                        "sample_rate": sample_rate,
                        "chunk_id": 1,
                        "time": time.time(),
                    }
                )
            elif audio_base64 is None and sample_rate is None:
                await self.audio_chunk_queue.put(None)

            await asyncio.sleep(0)
            await self.vae_idle_event.wait()

        except Exception as e:
            logger.exception(f"Failed to process message in ModelHandler: {e}")
            logger.exception(traceback.format_exc())

    async def process_audio(self, ready_event):
        logger.info("Starting audio processing task")
        ready_event.set()
        while True:
            audio = None
            audio_list = []
            audio_segment_id = 0
            chunk_count = 0
            try:
                while True:
                    await asyncio.sleep(0)
                    await self.vae_idle_event.wait()
                    chunk = await self.audio_chunk_queue.get()

                    if chunk is None:
                        break

                    await asyncio.sleep(0)
                    await self.vae_idle_event.wait()
                    current_audio_bytes = base64.b64decode(chunk["audio_base64"])
                    sample_rate = chunk["sample_rate"]
                    chunk_id = chunk["chunk_id"]
                    enqueue_time = chunk["time"]
                    logger.info("Audio chunk %d decoded" % chunk_id)

                    await asyncio.sleep(0)
                    await self.vae_idle_event.wait()
                    current_audio, sr = torchaudio.load(
                        current_audio_bytes, format="s16le"
                    )
                    current_audio = current_audio.to("cuda")

                    await asyncio.sleep(0)
                    await self.vae_idle_event.wait()
                    current_audio_16k = torchaudio.functional.resample(
                        current_audio,
                        orig_freq=sample_rate,
                        new_freq=service_config.audio.sample_rate,
                    )

                    if chunk_id == 0:
                        audio_list.append(
                            current_audio_16k[
                                ...,
                                int(
                                    self.audio_watermark_length
                                    * service_config.audio.sample_rate
                                ) :,
                            ]
                        )
                    else:
                        audio_list.append(current_audio_16k)

                    while sum([x.shape[-1] for x in audio_list]) >= (
                        self.audio_chunk_sizes[
                            min(audio_segment_id, len(self.audio_chunk_sizes) - 1)
                        ]
                        * service_config.audio_samples_per_video_block
                    ):
                        await asyncio.sleep(0)
                        await self.vae_idle_event.wait()
                        audio_segment_id += 1
                        tmp_audio = torch.cat(audio_list, dim=-1)
                        chunk_audio = tmp_audio[..., : self.audio_chunk_length]
                        audio = tmp_audio[..., self.audio_chunk_length :]
                        audio_list = [audio]

                        await asyncio.sleep(0)
                        await self.vae_idle_event.wait()
                        chunk_audio_base64 = encode_audio_to_base64(chunk_audio)

                        await self._process_audio_chunk(
                            {
                                "audio_base64": chunk_audio_base64,
                                "decoded_audio": chunk_audio,
                            }
                        )

                    chunk_count += 1  # audio chunk

                if sum([x.shape[-1] for x in audio_list]) > 0:
                    await asyncio.sleep(0)
                    await self.vae_idle_event.wait()
                    tmp_audio = torch.cat(audio_list, dim=-1)
                    audio_base64 = encode_audio_to_base64(tmp_audio)

                    await self._process_audio_chunk(
                        {"audio_base64": audio_base64, "decoded_audio": tmp_audio}
                    )

                logger.info("TTS processing done.")
                await self._process_audio_chunk(None)

            except Exception as e:
                logger.exception("Exception in process_audio: {e}")
                logger.exception(traceback.format_exc())

    async def _process_audio_chunk(self, audio_data):
        await asyncio.sleep(0)
        await self.vae_idle_event.wait()
        try:
            if audio_data is None:
                audio_base64 = None
                await self.lip_sync_manager.process_audio_chunk(None, None)
            else:
                audio_base64 = audio_data.get("audio_base64", "")
                decoded_audio = audio_data.get("decoded_audio", None)  # tensor
                if not audio_base64:
                    return
                if decoded_audio is None or len(decoded_audio) == 0:
                    return

                await self.lip_sync_manager.process_audio_chunk(
                    audio_base64, decoded_audio
                )

        except Exception as e:
            logger.exception(f"Failed to process audio chunk: {e}")
            raise
