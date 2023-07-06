import asyncio
import zmq.asyncio
import json
import logging
from typing import Dict
import zmq
from pprint import pformat

log = logging.getLogger(__name__)

async def send_request_async(
    request: Dict,
    ip: str = "127.0.0.1",
    port: str = "5555",
) -> Dict:
    log.info(f"Connecting to {ip}:{port} ...")
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{ip}:{port}")
    log.info("... Done!")
    log.info(f"Sending request: {request}")
    await socket.send_json(json.dumps(request))
    log.info("Waiting for response...")
    response = json.loads(await socket.recv_json())
    log.info(f"Received response: {pformat(response)}")
    return response

if __name__ == "__main__":
    asyncio.run(send_request_async({"message": "Hello, World!"}))