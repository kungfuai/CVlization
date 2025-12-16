import asyncio
import base64
import io
import logging

import cv2
import numpy as np
import torch
import torchaudio


def encode_image_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """Encode image to base64"""
    try:
        _, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        return base64.b64encode(buffer.tobytes()).decode()
    except Exception as e:
        logging.error(f"Failed to encode image: {e}")
        return ""


async def encode_image_async(*args, **kwargs):
    ret = encode_image_to_base64(*args, **kwargs)
    await asyncio.sleep(0)
    return ret


def encode_audio_to_base64(audio: torch.Tensor, sample_rate: int = 16000) -> str:
    audio_buffer = io.BytesIO()
    torchaudio.save(
        audio_buffer, src=audio.to("cpu"), sample_rate=sample_rate, format="wav"
    )
    audio_byte = audio_buffer.getvalue()
    audio_buffer.close()
    audio_base64 = base64.b64encode(audio_byte).decode("utf-8")
    return audio_base64
