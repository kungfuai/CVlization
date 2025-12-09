"""
FastAPI web server for OpenVLA + SimplerEnv demo.

Provides a WebSocket endpoint for streaming simulation frames
and a simple web UI for task selection.
"""

import os
import sys
import logging
import asyncio
import base64
import json
from pathlib import Path
from typing import Optional

# Suppress verbose logging before heavy imports
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("DISPLAY", "")  # Headless rendering
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from sim_runner import SimRunner, get_available_tasks, TASKS


# Load configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH) as f:
        CONFIG = yaml.safe_load(f)
else:
    CONFIG = {
        "model_path": "openvla/openvla-7b",
        "max_steps": 200,
        "frame_delay_ms": 100,  # Delay between frames in ms
        "jpeg_quality": 80,
    }


app = FastAPI(title="OpenVLA SimplerEnv Demo")

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>OpenVLA SimplerEnv Demo</h1><p>Static files not found.</p>")


@app.get("/api/tasks")
async def list_tasks():
    """List available tasks with metadata."""
    return {
        "tasks": [
            {
                "id": task_id,
                "description": info["description"],
                "embodiment": info["embodiment"],
                "control_freq": info["control_freq"],
            }
            for task_id, info in TASKS.items()
        ]
    }


@app.get("/api/config")
async def get_config():
    """Return current configuration."""
    return CONFIG


def encode_frame(image: np.ndarray, quality: int = 80) -> str:
    """Encode RGB image to base64 JPEG."""
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Encode as JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", bgr, encode_params)
    # Convert to base64
    return base64.b64encode(buffer.tobytes()).decode("ascii")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for streaming simulation frames.

    Protocol:
        Client sends: {"task_id": "...", "max_steps": N, "use_random": bool}
        Server sends: {"type": "frame", "step": N, "image": base64, "instruction": "..."}
        Server sends: {"type": "done", "success": bool, "steps": N}
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        # Wait for initialization message
        init_msg = await websocket.receive_text()
        config = json.loads(init_msg)

        task_id = config.get("task_id", "widowx_spoon_on_towel")
        max_steps = config.get("max_steps", CONFIG.get("max_steps", 200))
        use_random = config.get("use_random", False)
        model_path = CONFIG.get("model_path", "openvla/openvla-7b")

        logger.info(f"Starting episode: task={task_id}, max_steps={max_steps}, random={use_random}")

        # Send acknowledgment
        await websocket.send_text(json.dumps({
            "type": "init",
            "task_id": task_id,
            "max_steps": max_steps,
            "model_path": model_path if not use_random else "random",
        }))

        # Create simulation runner
        runner = SimRunner(
            task_id=task_id,
            model_path=model_path,
            max_steps=max_steps,
        )

        # Choose episode type
        if use_random:
            episode_gen = runner.get_random_action_episode()
        else:
            episode_gen = runner.run_episode()

        # Stream frames
        frame_delay = CONFIG.get("frame_delay_ms", 100) / 1000.0
        jpeg_quality = CONFIG.get("jpeg_quality", 80)
        final_success = False
        final_step = 0

        try:
            for result in episode_gen:
                # Encode frame
                image_b64 = encode_frame(result.image, quality=jpeg_quality)

                # Send frame message
                msg = {
                    "type": "frame",
                    "step": result.step,
                    "image": image_b64,
                    "instruction": result.instruction,
                    "reward": float(result.reward),
                    "success": result.success,
                }
                await websocket.send_text(json.dumps(msg))

                final_success = result.success
                final_step = result.step

                # Check for client disconnect
                try:
                    # Non-blocking check for close message
                    await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=0.001
                    )
                except asyncio.TimeoutError:
                    pass  # No message, continue
                except WebSocketDisconnect:
                    logger.info("Client disconnected during episode")
                    return

                # Add delay for viewing
                await asyncio.sleep(frame_delay)

        except StopIteration as e:
            # Episode finished, get return value
            if e.value:
                final_success, _ = e.value

        # Send completion message
        await websocket.send_text(json.dumps({
            "type": "done",
            "success": final_success,
            "steps": final_step,
        }))

        logger.info(f"Episode complete: success={final_success}, steps={final_step}")

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e),
            }))
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


def main():
    """Run the server."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="OpenVLA SimplerEnv Demo Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
