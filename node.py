from abc import ABC, abstractmethod
import zmq
import asyncio
import json
from datetime import datetime, timedelta

import asyncio
import zmq.asyncio
import json
from abc import ABC, abstractmethod

class LeafNode(ABC):
    def __init__(self, ip="127.0.0.1", port="5555"):
        self.ip = ip
        self.port = port
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://{self.ip}:{self.port}")
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, input):
        pass

    async def miniserver(self, idle_timeout=30):
        while True:
            self.socket.setsockopt(zmq.RCVTIMEO, idle_timeout * 1000)
            try:
                print("Waiting for a request...")
                message = await self.socket.recv_json()
                request = json.loads(message)
                print(f"Received request: {request}")
                response = self.predict(request)
                print(f"Sending response: {response}")
                await self.socket.send_json(json.dumps(response))
            except zmq.error.Again:
                print("Idle timeout reached. Closing server.")
                break
            except Exception as e:
                print(f"Error occurred: {e}")
                response = {"error": f"Error occurred: {e}"}
                await self.socket.send_json(json.dumps(response))
        self.socket.close()

class MasterNode:
    def __init__(self, leaf_addresses):
        """
        :param leaf_addresses: A list of tuples containing the (ip, port) of each leaf node.
        """
        self.context = zmq.asyncio.Context()
        self.leaf_addresses = leaf_addresses

    async def send_request(self, address, request):
        """
        Send a request to a single leaf node and return the response.
        :param address: The (ip, port) of the leaf node.
        :param request: The request to send.
        :return: The response from the leaf node.
        """
        ip, port = address
        socket = self.context.socket(zmq.REQ)
        socket.connect(f"tcp://{ip}:{port}")
        await socket.send_json(json.dumps(request))
        response = await socket.recv_json()
        socket.close()
        return response

    async def gather_responses(self, request):
        """
        Send a request to all leaf nodes and return their responses.
        :param request: The request to send.
        :return: A list of responses from all leaf nodes.
        """
        tasks = [asyncio.create_task(self.send_request(address, request)) for address in self.leaf_addresses]
        responses = await asyncio.gather(*tasks)
        return responses

    def run(self, request):
        responses = asyncio.run(self.gather_responses(request))
        return responses