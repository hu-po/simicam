"""

https://github.com/chaoningzhang/mobilesam#installation

"""

import asyncio
import json
import logging
from pprint import pformat

import cv2
import torch
import zmq
import zmq.asyncio
from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

log = logging.getLogger(__name__)


async def receive_request_async(ip: str = "127.0.0.1", port: str = "5555") -> None:
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


def model_inference(
    model="vit_t",
    checkpoint="./weights/mobile_sam.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if isinstance(model, str):
        model = sam_model_registry[model](checkpoint=checkpoint)
        model.to(device=device)
        model.eval()

    TEST_IMAGE_FILEPATH = "/home/oop/Downloads/car.jpeg"
    image = cv2.imread(TEST_IMAGE_FILEPATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # With prompts
    # predictor = SamPredictor(model)
    # predictor.set_image(<your_image>)
    # masks, _, _ = predictor.predict(<input_prompts>)

    # Entire image at once
    mask_generator = SamAutomaticMaskGenerator(model)
    mask_generator.generate(image)

    return model


if __name__ == "__main__":
    BIND_URI = "tcp://*:5555"
    
    # Server Side
    asyncio.run(receive_request_async())
