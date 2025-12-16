import asyncio
import json
import logging
import socket
import traceback
import uuid
from queue import Queue

from self_forcing.utils import parallel_state as mpu

logger = logging.getLogger(__name__)


def run_socket_server(queue: Queue, port, host="localhost"):
    assert port is not None
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_socket.bind((host, port))

    server_socket.listen(1)
    logger.info(f"Rank {mpu.get_rank()}: Server listening on port {port}")

    client_socket, addr = server_socket.accept()
    buffer = b""
    try:
        while True:
            data = client_socket.recv(256)
            buffer += data
            pos = buffer.find(b"\n")
            while pos >= 0:
                try:
                    line = buffer[: pos + 1].strip()
                    buffer = buffer[pos + 1 :]
                    pos = buffer.find(b"\n")

                    id = str(uuid.uuid4())
                    msg = line.decode("utf-8")
                    logger.info(f"Rank {mpu.get_rank()}, Socket server received: {msg}")
                    data_dict = json.loads(msg)

                    queue.put(data_dict, block=True)

                except json.JSONDecodeError:
                    logger.exception(f"Invalid JSON received: {msg}")
    except Exception as e:
        logger.exception(f"Exception in socket server: {e}")


async def run_socket_server_async(
    queue: asyncio.Queue, port, server_ready_event: asyncio.Event, host="localhost"
):
    assert port is not None

    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, queue), host, port
    )
    logger.info("Rank %d: Server listening on port %d" % (mpu.get_rank(), port))
    server_ready_event.set()

    async with server:
        await server.serve_forever()


async def handle_client(reader, writer, queue):
    addr = writer.get_extra_info("peername")
    logger.info(f"Async socket server connection from {addr}")

    try:
        buffer = b""
        while True:
            data = await reader.read(256)
            if not data:
                continue

            buffer += data
            pos = buffer.find(b"\n")
            while pos >= 0:
                line = buffer[: pos + 1].strip()
                buffer = buffer[pos + 1 :]
                pos = buffer.find(b"\n")

                message = line.decode("utf-8")
                logger.info(f"Async socket server received: {message}")

                try:
                    data_dict = json.loads(message)
                    await queue.put(data_dict)
                except json.JSONDecodeError:
                    logger.exception(f"Invalid JSON received: {message}")

    except Exception as e:
        logger.exception(f"Error handling client {addr}: {e}")
    finally:
        writer.close()
        await writer.wait_closed()
        logger.info(f"Connection closed with {addr}")


def socket_send(data: dict, port, host="localhost", client_socket=None):
    if client_socket is None:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        client_socket.connect((host, port))

    client_socket.send(json.dumps(data, ensure_ascii=False).encode("utf-8") + b"\n")
    return client_socket
