import asyncio
import json
import threading
import websockets
from typing import Set, Tuple, Optional

from simulator.models.observer import SimulationObserver
from simulator.models.messages import (
    BaseMessage,
    SimulationInitializedMessage,
    ObstaclesProcessedMessage,
    AgentPositionsUpdateMessage,
    GoalsProcessedMessage
)

class WebSocketRenderer(SimulationObserver):
    def __init__(self, host="localhost", port=8080):
        self.host = host
        self.port = port
        self.active = True
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.loop = asyncio.new_event_loop()

        # Start the websocket server in a background thread
        self.server_thread = threading.Thread(target=self._start_server, daemon=True)
        self.server_thread.start()

    def _make_handler(self):
        async def handler(websocket):
            self.clients.add(websocket)
            print("✅ Client connected")
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


        # Schedule the coroutine without blocking the thread
        self.loop.create_task(run_server())

        # Now start the loop — this never returns (unless loop is stopped)
        self.loop.run_forever()

    def setup(self):
        print("WebSocketRenderer setup complete.")

    def is_active(self):
        return self.active

    def stop(self):
        self.active = False
        self.loop.call_soon_threadsafe(self.loop.stop)

    def update(self, message: BaseMessage):
        print(f"[WebSocketRenderer] Received message: {type(message)}")
        if isinstance(message, SimulationInitializedMessage):
            data = {
                "type": "simulation_initialized",
                "agents": message.agent_initialization_data
            }
            self._send_to_clients(data)

        elif isinstance(message, ObstaclesProcessedMessage):
            data = {
                "type": "obstacles_processed",
                "obstacles": message.obstacles
            }
            self._send_to_clients(data)

        elif isinstance(message, GoalsProcessedMessage):
            data = {
                "type": "goals_processed",
                "goals": message.goals
            }
            self._send_to_clients(data)

        elif isinstance(message, AgentPositionsUpdateMessage):
            agents = []
            for agent_data in message.agent_positions:
                agent_id, x, y, velocity, pref_velocity, distance = agent_data
                agents.append({
                    "id": agent_id,
                    "position": [x, y],
                    "velocity": velocity,
                    "preferred_velocity": pref_velocity,
                    "distance_to_goal": distance
                })
            data = {
                "type": "agent_update",
                "step": message.step,
                "agents": agents
            }
            self._send_to_clients(data)

    def _send_to_clients(self, data: dict):
        if not self.clients:
            return
        json_data = json.dumps(data)
        asyncio.run_coroutine_threadsafe(self._broadcast(json_data), self.loop)

    async def _broadcast(self, message: str):
        to_remove = set()
        for client in self.clients:
            try:
                asyncio.create_task(client.send(message))
            except websockets.exceptions.ConnectionClosed:
                to_remove.add(client)
        self.clients -= to_remove
