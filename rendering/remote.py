import asyncio
import threading
import websockets
from typing import Set

from simulator.models.observer import SimulationObserver
from simulator.models.communication.messages import BaseMessage
from simulator.models.communication.protobuf.converters import to_protobuf_and_type

class WebSocketRenderer(SimulationObserver):
    def __init__(self, host="localhost", port=8080):
        self.host = host
        self.port = port
        self.active = True
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.loop = asyncio.new_event_loop()
        self.message_log: list[BaseMessage] = []  # Store all messages for replay

        # Start the websocket server in a background thread
        self.server_thread = threading.Thread(target=self._start_server, daemon=True)
        self.server_thread.start()

    def _make_handler(self):
        async def handler(websocket):
            self.clients.add(websocket)
            print("✅ Client connected")

            # Send buffered history
            for msg in self.message_log:
                msg_type, binary = to_protobuf_and_type(msg)
                header = f"{msg_type}\n".encode("utf-8")
                try:
                    await websocket.send(header + binary)
                except websockets.exceptions.ConnectionClosed:
                    print("❌ Failed to send history message")
                    break

            try:
                async for _ in websocket:
                    pass
            except:
                print("❌ Client disconnected")
            finally:
                self.clients.remove(websocket)
        return handler

    def _start_server(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def run_server():
            handler = self._make_handler()
            self.server = await websockets.serve(handler, self.host, self.port)
            print(f"🌐 WebSocketRenderer running on ws://{self.host}:{self.port}")

        self.loop.create_task(run_server())
        self.loop.run_forever()

    def setup(self):
        print("WebSocketRenderer setup complete.")

    def is_active(self):
        return self.active

    def stop(self):
        self.active = False
        self.loop.call_soon_threadsafe(self.loop.stop)

    def update(self, message: BaseMessage):
        # print(f"[WebSocketRenderer] Received message: {type(message)}")
        self.message_log.append(message)
        msg_type, binary = to_protobuf_and_type(message)
        header = f"{msg_type}\n".encode("utf-8")
        asyncio.run_coroutine_threadsafe(self._broadcast(header + binary), self.loop)

    async def _broadcast(self, message: bytes):
        to_remove = set()
        for client in self.clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                to_remove.add(client)
        self.clients -= to_remove
