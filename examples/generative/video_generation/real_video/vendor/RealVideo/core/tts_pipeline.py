import asyncio
import logging
import os
import re
import time

import aiohttp
import orjson

logger = logging.getLogger(__name__)


class TTSPipeline:
    def __init__(
        self,
        vae_idle_event: asyncio.Event,
        model_name_llm="glm-4.5-airx",
        model_name_tts="glm-tts",
    ):
        self.vae_idle_event = vae_idle_event

        self.model_name_llm = model_name_llm
        self.model_name_tts = model_name_tts
        self.stc_split_pattern = r"([。？！?!\n]”?|[.?!]\s?)"
        self.substc_split_pattern = "(，|, )"
        self.stc_min_length = 10
        self.stc_max_length = 50
        self.chat_history = []
        self.async_tasks_started = False
        self.proxy = os.environ.get("HTTP_PROXY", None) or os.environ.get(
            "http_proxy", None
        )
        self.llm_task = None
        self.tts_task = None

    def reset_status(self):
        self.chat_history = []

    def start_async_tasks(
        self, text_input_queue: asyncio.Queue, output_queue: asyncio.Queue
    ):
        if not self.async_tasks_started:
            sentence_queue = asyncio.Queue(32)
            self.llm_task = asyncio.create_task(
                self.llm_worker_async(text_input_queue, sentence_queue)
            )
            self.tts_task = asyncio.create_task(
                self.tts_worker_async(sentence_queue, output_queue)
            )
            self.async_tasks_started = True
            logger.info("LLM & TTS tasks created")

    async def llm_worker_async(self, text_input_queue: asyncio.Queue, sentence_queue):
        headers = {
            "Authorization": f"Bearer {os.environ['ZAI_API_KEY']}",
            "Content-Type": "application/json",
        }
        llm_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        body_template = {
            "model": self.model_name_llm,
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": True,
            "thinking": {"type": "disabled"},
        }

        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    text_item = await text_input_queue.get()
                    profile = text_item.get("profile", None)
                    text_input = text_item["text"]
                    voice_id = text_item.get("voice_id", None)
                    logger.info(f"LLM processing: {text_item}")

                    body = {
                        "messages": [{"role": "system", "content": profile}]
                        + self.chat_history
                        + [{"role": "user", "content": text_input}]
                    }
                    body.update(body_template)

                    buffer = b""
                    chunk_resp = b""
                    text_buffer = ""
                    finished = False
                    text_response = ""
                    current_sentence = ""

                    start = time.time()

                    logger.info(f"Creating LLM stream response for input: {text_input}")
                    async with session.post(
                        llm_url, headers=headers, json=body, proxy=self.proxy
                    ) as response:
                        logger.info(
                            "LLM stream response for input %s created, %.3fms elapsed"
                            % (text_input, 1000 * (time.time() - start))
                        )

                        while True:
                            await asyncio.sleep(0)
                            await self.vae_idle_event.wait()
                            chunk_resp = await response.content.readline()

                            buffer += chunk_resp
                            if not buffer:
                                break

                            pos = buffer.find(b"\n")
                            if pos == -1 and not chunk_resp:
                                pos = len(buffer)

                            while pos > -1:
                                await asyncio.sleep(0)
                                await self.vae_idle_event.wait()

                                bline = buffer[: pos + 1]
                                buffer = buffer[pos + 1 :]
                                pos = buffer.find(b"\n")

                                if not bline:
                                    break

                                bline = bline.strip()
                                if not bline or not bline.startswith(b"data:"):
                                    continue

                                if bline.startswith(b"data: [DONE]"):
                                    break

                                await asyncio.sleep(0)
                                await self.vae_idle_event.wait()
                                chunk = orjson.loads(
                                    bline[6:].strip()
                                )  # remove 'data: '

                                finished = chunk["choices"][0].get("finish_reason", "")
                                if finished == "stop" or finished == "stop_sequence":
                                    break

                                if chunk["choices"][0]["delta"]["content"]:
                                    text_chunk = chunk["choices"][0]["delta"]["content"]
                                    text_buffer += text_chunk

                                    while True:
                                        await asyncio.sleep(0)
                                        await self.vae_idle_event.wait()
                                        m = re.search(
                                            self.stc_split_pattern, text_buffer
                                        )
                                        if m is not None:
                                            if m.end() > self.stc_max_length:
                                                pass

                                            current_sentence = text_buffer[
                                                : m.end()
                                            ].strip()
                                            text_buffer = text_buffer[m.end() :]
                                            if current_sentence:
                                                await sentence_queue.put(
                                                    {
                                                        "sentence": current_sentence,
                                                        "voice_id": voice_id,
                                                    }
                                                )
                                                logger.info(
                                                    "LLM current_sentence: %s time used: %.3fms"
                                                    % (
                                                        current_sentence,
                                                        ((time.time() - start) * 1000),
                                                    )
                                                )
                                                text_response += current_sentence

                                        else:
                                            break

                    text_buffer = text_buffer.strip()
                    if text_buffer:
                        await sentence_queue.put(
                            {"sentence": text_buffer, "voice_id": voice_id}
                        )
                        text_response += text_buffer

                    await sentence_queue.put(None)
                    self.chat_history.append(
                        {"role": "assistant", "content": text_response}
                    )

                    await asyncio.sleep(0)
                    await self.vae_idle_event.wait()
                except Exception as e:
                    logger.exception(f"Exception in LLM worker: {e}")

    async def tts_worker_async(
        self, sentence_queue: asyncio.Queue, output_queue: asyncio.Queue
    ):
        headers = {
            "Authorization": f"Bearer {os.environ['ZAI_API_KEY']}",
            "Content-Type": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=10)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                try:
                    sentence_item = await sentence_queue.get()
                    if sentence_item is None:  # llm response finish
                        logger.info("TTS %s done" % str(sentence))
                        await output_queue.put(None)
                        continue

                    sentence = sentence_item.get("sentence", None)
                    voice_id = sentence_item.get("voice_id", None)
                    logger.info(f"TTS processing: {sentence_item}")

                    body = {
                        "input": sentence,
                        "stream": True,
                        "model": "glm-tts",
                        "voice": voice_id,
                        "response_format": "pcm",
                        "speed": 1.0,
                        "volume": 1.0,
                    }
                    tts_url = "https://open.bigmodel.cn/api/paas/v4/audio/speech"

                    buffer = b""
                    chunk_id = -1
                    finished = False
                    chunk_resp_list = []
                    chunk_resp_list_length = 0

                    start = time.time()
                    logger.info(f"Creating TTS stream for sentence: {sentence}")
                    async with session.post(
                        tts_url, headers=headers, json=body, proxy=self.proxy
                    ) as response:
                        logger.info(
                            "TTS stream response for %s created, %.3fms elapsed"
                            % (sentence, 1000 * (time.time() - start))
                        )

                        while True:
                            await asyncio.sleep(0)
                            await self.vae_idle_event.wait()
                            chunk_resp = await response.content.read(1024)

                            await asyncio.sleep(0)
                            await self.vae_idle_event.wait()

                            pos = chunk_resp.find(b"\n")
                            if pos > -1:
                                pos += chunk_resp_list_length

                            chunk_resp_list.append(chunk_resp)
                            chunk_resp_list_length += len(chunk_resp)
                            if chunk_resp_list_length == 0:
                                break

                            if pos == -1 and not chunk_resp:
                                pos = chunk_resp_list_length

                            while pos > -1:
                                await asyncio.sleep(0)
                                await self.vae_idle_event.wait()
                                buffer = b"".join(chunk_resp_list)

                                bline = buffer[: pos + 1]
                                buffer = buffer[pos + 1 :]
                                chunk_resp_list = [buffer]
                                chunk_resp_list_length = len(buffer)
                                pos = buffer.find(b"\n")
                                chunk_id += 1

                                logger.info("Processing audio chunk %d" % chunk_id)

                                await asyncio.sleep(0)
                                await self.vae_idle_event.wait()
                                bline = bline.strip()
                                if not bline:
                                    break

                                if not bline or not bline.startswith(b"data:"):
                                    continue

                                await asyncio.sleep(0)
                                await self.vae_idle_event.wait()
                                chunk = orjson.loads(bline[5:])  # remove 'data:'

                                choice = chunk["choices"][0]
                                index = choice["index"]
                                is_finished = choice.get("finish_reason", "")
                                if is_finished == "stop":
                                    finished = True
                                    break
                                audio_delta = choice["delta"]["content"]
                                sr = choice["delta"]["return_sample_rate"]

                                logger.info(
                                    f"TTS stream: {index}.audio_delta={audio_delta[:64]}..., length={len(audio_delta)}"
                                )
                                await output_queue.put(
                                    {
                                        "audio_base64": audio_delta,
                                        "sample_rate": sr,
                                        "chunk_id": chunk_id,
                                        "time": time.time(),
                                    }
                                )

                            if finished:
                                break

                except Exception as e:
                    logger.exception(f"Exception in TTS worker: {e}")
