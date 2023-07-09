import gc
import json
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable, Dict
from pprint import pformat

import torch
import zmq
import zmq.asyncio
import time

log = logging.getLogger(__name__)

def get_device(device: str = None):
    if device is None or device == "gpu" or device.startswith("cuda"):
        if torch.cuda.is_available():
            print("Using GPU")
            print("Clearing GPU memory")
            torch.cuda.empty_cache()
            gc.collect()
            if device.startswith("cuda"):
                return torch.device(device)
            return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

def time_and_log(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        log.debug(f"Calling: {func.__name__} - duration: {duration:.2f}")
        return result

    return wrapper

async def microservice_server(
    ip: str = "127.0.0.1",
    port: str = "5555",
    timeout: timedelta = timedelta(seconds=30),
    init_func: Callable = None,
    loop_func: Callable = None,
    **kwargs,
):
    log.info(f"Listening on {ip}:{port} ...")
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{ip}:{port}")
    _init_output: Dict = init_func(**kwargs)
    start_time: datetime = datetime.now()
    log.info(f"Starting server, timeout: {timeout}")
    while True:
        if (datetime.now() - start_time) > timeout:
            log.info("Timeout reached. Closing server.")
            return
        log.info("Waiting for a request...")
        message = await socket.recv_json()
        request: Dict = json.loads(message)
        log.info(f"Received request: {pformat(request)}")
        response = loop_func(request=request, **_init_output, **kwargs)
        log.info(f"Sending response: {pformat(response)}")
        await socket.send_json(json.dumps(response))


async def microservice_client(
    ip: str = "127.0.0.1",
    port: str = "5555",
    timeout: timedelta = timedelta(seconds=30),
    request_func: Callable = None,
    **kwargs,
) -> Dict:
    log.info(f"Connecting to {ip}:{port} ...")
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{ip}:{port}")
    start_time: datetime = datetime.now()
    log.info(f"Starting client, timeout: {timeout}")
    while True:
        if (datetime.now() - start_time) > timeout:
            log.info("Timeout reached. Closing client.")
            return
        request: Dict = request_func(**kwargs)
        log.info(f"Sending request: {request}")
        await socket.send_json(json.dumps(request))
        log.info("Waiting for response...")
        response = json.loads(await socket.recv_json())
        log.info(f"Received response: {pformat(response)}")
