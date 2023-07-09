"""

https://github.com/chaoningzhang/mobilesam#installation

"""

import asyncio
import json
import logging
import gc
from pprint import pformat

import numpy as np
import torch
import zmq
import zmq.asyncio
from mobile_sam import (SamAutomaticMaskGenerator, SamPredictor,
                        sam_model_registry)
from PIL import Image

log = logging.getLogger(__name__)


async def receive_request_async(ip: str = "127.0.0.1", port: str = "5556") -> None:
    log.info(f"Listening on {ip}:{port} ...")
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{ip}:{port}")
    while True:
        log.info("Waiting for a request...")
        message = await socket.recv_json()
        request = json.loads(message)
        log.info(f"Received request: {pformat(request)}")

        if "message" in request:
            response = {"response": request["message"][::-1]}
        else:
            response = {"error": 'No "message" field in request'}

        log.info(f"Sending response: {pformat(response)}")
        await socket.send_json(json.dumps(response))

def get_device(device: str = None):
    if device == None or device == "gpu":
        if torch.cuda.is_available():
            print("Using GPU")
            print("Clearing GPU memory")
            torch.cuda.empty_cache()
            gc.collect()
            return torch.device("cuda")
    print("Using CPU")
    return torch.device("cpu")

def load_model(
    model="vit_t",
    checkpoint="./weights/mobile_sam.pt",
    device=None,
):
    device = get_device(device)
    if isinstance(model, str):
        model = sam_model_registry[model](checkpoint=checkpoint)
        model.to(device=device)
        model.eval()
    return model

def get_masks(
    image: np.ndarray = None,
    model=None,
    prompts=None,
):
    if prompts:
        print(f"Using prompts {prompts}")
        predictor = SamPredictor(model)
        predictor.set_image(image)
        masks, _, _ = predictor.predict(prompts)
    else:
        print("Using automatic mask generation")
        mask_generator = SamAutomaticMaskGenerator(model)
        image = np.array(Image.open('data/test.png'))
        masks = mask_generator.generate(image)
    print(f"Found {len(masks)} masks: {masks}")
    return masks


if __name__ == "__main__":
    BIND_URI = "tcp://*:5556"

    model = load_model()
    image = np.array(Image.open('data/test.png'))
    masks = get_masks(model=model, prompts=None)

    model = load_model()
    image = np.array(Image.open('data/test.png'))
    masks = get_masks(model=model, prompts=None)
    
    # Server Side
    asyncio.run(receive_request_async())
