import json
import logging
from typing import Dict, List

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[int, WebSocket] = {}
        self.connection_count = 0

    async def connect(self, websocket: WebSocket, client_id: int = None):
        if client_id is None:
            client_id = self.connection_count
            self.connection_count += 1

        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

        await self.broadcast_json(
            {
                "type": "connection_status",
                "message": f"Client {client_id} connected",
                "client_id": client_id,
                "total_connections": len(self.active_connections),
            }
        )

        return client_id

    def disconnect(self, client_id: int):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")

    async def broadcast(self, message: str):
        try:
            disconnected_clients = []

            for client_id, connection in self.active_connections.items():
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Broadcast to client {client_id} failed: {e}")
                    disconnected_clients.append(client_id)

            for client_id in disconnected_clients:
                self.disconnect(client_id)
        except Exception as e:
            logger.exception("Exception in broadcast:", e, type(e), flush=True)

    async def broadcast_json(self, data: dict):
        await self.broadcast(json.dumps(data))

    def get_connection_count(self) -> int:
        return len(self.active_connections)

    def get_client_ids(self) -> List[int]:
        return list(self.active_connections.keys())

    def is_connected(self, client_id: int) -> bool:
        return client_id in self.active_connections

    async def send_system_status(self):
        status_data = {
            "type": "system_status",
            "total_connections": len(self.active_connections),
            "client_ids": list(self.active_connections.keys()),
            "timestamp": None,
        }
        await self.broadcast_json(status_data)
