import json
import logging
import os

import requests

logger = logging.getLogger(__name__)


def get_voice_list():
    try:
        url = "https://open.bigmodel.cn/api/paas/v4/voice/list"
        headers = {"Authorization": f"Bearer {os.environ['ZAI_API_KEY']}"}
        response = requests.get(url, headers=headers)

        ret = json.loads(response.text)["voice_list"]
        voice_list = [(x["voice_name"], x["voice"]) for x in ret]
        return voice_list

    except Exception as e:
        logger.exception(f"Exception in get_voice_list: {e}")
        return []


def upload_audio_file(file_path):
    try:
        url = "https://open.bigmodel.cn/api/paas/v4/files"
        files = {"file": (os.path.basename(file_path), open(file_path, "rb"))}
        payload = {"purpose": "voice-clone-input"}
        headers = {"Authorization": f"Bearer {os.environ['ZAI_API_KEY']}"}

        response = requests.post(url, data=payload, files=files, headers=headers)

        ret = json.loads(response.text)
        id = ret["id"]
        return id
    except Exception as e:
        logger.exception(f"Exception in upload_audio: {e}")
        raise


def clone(file_id, voice_name):
    try:
        url = "https://open.bigmodel.cn/api/paas/v4/voice/clone"
        payload = {
            "model": "glm-tts-clone",
            "voice_name": voice_name,
            "input": "Hello world.",
            "file_id": file_id,
        }
        headers = {
            "Authorization": f"Bearer {os.environ['ZAI_API_KEY']}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=payload, headers=headers)

        return json.loads(response.text)

    except Exception as e:
        logger.exception(f"Exception in voice clone: {e}")
        raise
