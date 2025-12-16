import asyncio
import json
import logging
import os

import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger(__name__)

import logging
import time
import traceback
import uuid

from config.config import config

from .connection import ConnectionManager
from .model_handler import ModelHandler
from .voice_clone import clone, get_voice_list, upload_audio_file


class RealVideoApp:
    def __init__(self):
        self.app = FastAPI(title="RealVideo")
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.model_handler = ModelHandler()
        self.connection_manager = ConnectionManager()
        self.lip_sync_manager = self.model_handler.lip_sync_manager

        self.upload_folder = "uploads"
        self.allowed_image_exts = {"png", "jpg", "jpeg"}
        self.allowed_audio_exts = {"mp3", "wav"}
        self.max_file_size = 10 * 1024 * 1024
        self.last_ws_message_time = None
        self.ws_lifecheck_task = None

        self._setup_routes()

        logger.info("Initialization finished.")

    def allowed_file(self, filename, file_type="img"):
        return (
            "." in filename
            and filename.rsplit(".", 1)[1].lower() in self.allowed_image_exts
            if file_type == "img"
            else self.allowed_audio_exts
        )

    def _setup_routes(self):
        @self.app.get("/", response_class=HTMLResponse)
        async def get_homepage():
            try:
                with open("templates/index.html", "r", encoding="utf-8") as f:
                    return HTMLResponse(content=f.read())
            except Exception as e:
                logger.exception(f"Failed to load homepage: {e}")
                return HTMLResponse(content="<h1>Server error</h1>")

        @self.app.get("/api/status")
        async def get_system_status():
            return {
                "status": "running",
                "connections": self.connection_manager.get_connection_count(),
                "timestamp": time.time(),
            }

        @self.app.post("/upload_image")
        async def upload_image(image: UploadFile = File(...)):
            os.makedirs(self.upload_folder, exist_ok=True)
            if not self.allowed_file(image.filename):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported filetype. Available filetypes: "
                    + ", ".join(self.allowed_image_exts),
                )

            contents = await image.read()
            if len(contents) > self.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File is too large, maximum allowed is {self.max_file_size // (1024 * 1024)}MB",
                )

            file_extension = image.filename.split(".")[-1]
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = os.path.join(self.upload_folder, unique_filename)

            with open(file_path, "wb") as f:
                f.write(contents)

            return JSONResponse(
                {
                    "success": True,
                    "message": "Image uploaded",
                    "image_path": file_path,
                    "filename": unique_filename,
                }
            )

        @self.app.post("/upload_audio")
        async def upload_audio(
            audio: UploadFile = File(...),
        ):  # Upload wav and clone voice
            os.makedirs(self.upload_folder, exist_ok=True)
            if not self.allowed_file(audio.filename, file_type="audio"):
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported filetype. Available filetypes: "
                    + ", ".join(self.allowed_audio_exts),
                )

            contents = await audio.read()
            if len(contents) > self.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File is too large, maximum allowed is {self.max_file_size // (1024 * 1024)}MB",
                )

            file_extension = audio.filename.split(".")[-1]
            voice_name = os.path.splitext(os.path.basename(audio.filename))[0]
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = os.path.join(self.upload_folder, unique_filename)

            with open(file_path, "wb") as f:
                f.write(contents)
            logger.info(f"{file_path} saved")

            try:
                file_id = upload_audio_file(file_path)
                logger.info(f"file uploaded: {file_id}")
                clone_ret = clone(file_id, voice_name)
                logger.info(f"voice clone finished")

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error: {e}")

            voice_list = get_voice_list()
            return JSONResponse(
                {
                    "success": True,
                    "message": "Voice clone succeeded.",
                    "voice_list": voice_list,
                }
            )

        @self.app.get("/get_voice_list", response_class=JSONResponse)
        async def return_voice_list():
            voice_list = get_voice_list()
            return JSONResponse(
                {
                    "success": True,
                    "message": "Voice list fetched.",
                    "voice_list": voice_list,
                }
            )

        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket: WebSocket, client_id: int):
            logger.info(f"Connecting to websocket client {client_id}")

            try:
                if self.lip_sync_manager.websocket is not None:
                    await websocket.close()
                    logger.info(
                        "Active websocket exists, rejecting new websocket connection."
                    )
                    return

                else:
                    await self.connection_manager.connect(websocket, client_id)

                await self.lip_sync_manager.connect_websocket(websocket)
                await self.model_handler.start_jobs(websocket)

                self.model_handler.tts_pipeline.reset_status()

                if self.ws_lifecheck_task is None:
                    self.ws_lifecheck_task = asyncio.create_task(self.ws_lifecheck())
                    logger.info("WebSocket: ws_lifecheck task created")

                await self._handle_websocket_connection(websocket, client_id)

            except WebSocketDisconnect:
                await self.lip_sync_manager.disconnect_websocket()
                self.connection_manager.disconnect(client_id)
                logger.info(f"Client {client_id} disconnected")

            except Exception as e:
                await self.lip_sync_manager.disconnect_websocket()
                self.connection_manager.disconnect(client_id)

                logger.exception(f"Exception in Client {client_id}: {e}")
                logger.exception(traceback.format_exc())

    async def _handle_websocket_connection(self, websocket: WebSocket, client_id: int):
        while True:
            try:
                data = await websocket.receive_text()
                self.last_ws_message_time = time.time()
                message_data = json.loads(data)
                logger.info(message_data)

                logger.debug(
                    f"Received message from client {client_id}: {message_data.get('type', 'unknown')}"
                )

                if message_data["type"] in {"text", "audio"}:
                    await self._handle_text_audio_message(
                        message_data, websocket, client_id
                    )
                elif message_data["type"] == "ping":
                    await self._handle_ping_message(websocket, client_id)
                elif message_data["type"] in {"control", "image_config"}:
                    await self._handle_control_message(
                        message_data, websocket, client_id
                    )
                else:
                    logger.warning(f"Unknown message type: {message_data['type']}")

            except Exception as e:
                logger.error(
                    f"Failed to process message in _handle_websocket_connection: {e}, {type(e)}"
                )
                print(traceback.format_exc(), flush=True)
                await self.lip_sync_manager.disconnect_websocket()
                raise

    async def _handle_text_audio_message(
        self, message_data: dict, websocket: WebSocket, client_id: int
    ):
        if message_data["type"] == "text":
            profile_content = message_data.get("profile", "")
            text_content = message_data.get("text", "")
            audio_content = None
            sample_rate = None
            voice_id = message_data.get("voice_id", None)
            logger.info(
                f"Text message from client {client_id}: profile: {profile_content}, text: {text_content}"
            )

        elif message_data["type"] == "audio":
            profile_content = None
            text_content = None
            audio_content = message_data.get("audio", None)
            sample_rate = message_data.get("sample_rate", None)
            voice_id = None
            if audio_content is not None:
                logger.info(
                    f"Audio message from client {client_id}: {len(audio_content)}"
                )
            else:
                logger.info(f"Empty audio message from client {client_id}")

        processing_data = {
            "type": "processing_status",
            "status": "processing",
            "message": "Processing message...",
            "timestamp": message_data.get("timestamp", ""),
        }
        await websocket.send_text(json.dumps(processing_data))

        await self.model_handler.process_message(
            profile_content=profile_content,
            text_content=text_content,
            audio_base64=audio_content,
            sample_rate=sample_rate,
            voice_id=voice_id,
            websocket=websocket,
        )

    async def _handle_ping_message(self, websocket: WebSocket, client_id: int):
        pong_data = {"type": "pong", "timestamp": time.time(), "client_id": client_id}
        await websocket.send_text(json.dumps(pong_data))

    async def _handle_control_message(
        self, message_data, websocket: WebSocket, client_id: int
    ):
        try:
            logger.info(f"Control message from client {client_id}: {message_data}")
            await self.lip_sync_manager.process_control_message(message_data)

        except Exception as e:
            logger.warning(
                f"Failed to process control message in _handle_control_message: {e}, {type(e)}"
            )
            error_data = {
                "type": "error",
                "message": f"Failed to process control message, {e}",
                "timestamp": time.time(),
            }
            await websocket.send_text(json.dumps(error_data))
            raise

    async def ws_lifecheck(self):
        logger.info("entering ws lifecheck")
        while True:
            try:
                await asyncio.sleep(20)
                logger.info("checking websocket life")
                if (
                    self.last_ws_message_time is not None
                    and time.time() - self.last_ws_message_time > 60
                    and self.lip_sync_manager.websocket is not None
                ):
                    logger.info(
                        "Disconnecting websocket due to long time inactive %.3fs."
                        % (time.time() - self.last_ws_message_time)
                    )
                    await self.lip_sync_manager.disconnect_websocket()
                    self.lip_sync_manager.websocket = None

                elif (
                    self.lip_sync_manager.websocket is not None
                    and self.last_ws_message_time is not None
                ):
                    logger.info(
                        "Websocket lifecheck passed, %.3fs"
                        % (time.time() - self.last_ws_message_time)
                    )

                else:
                    logger.info("Websocket lifechecking, no active websocket")

            except Exception as e:
                logger.exception(f"Exception in ws_lifecheck: {e}")

    def run(self):
        logger.info(
            f"Starting server: http://{config.server.host}:{config.server.port}"
        )
        logger.info(config)

        uvicorn.run(
            self.app, host=config.server.host, port=config.server.port, log_level="info"
        )


def main():
    app = RealVideoApp()
    app.run()


if __name__ == "__main__":
    main()
