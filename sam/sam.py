"""

https://github.com/chaoningzhang/mobilesam#installation

"""

import json
import logging
from pprint import pformat
from typing import Dict

import cv2
import torch
import zmq
from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

log = logging.getLogger(__name__)


def load_model(
    model_type="vit_t",
    checkpoint="./weights/mobile_sam.pt",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()
    return model


def send_request(
    request: Dict,
    ip: str = "127.0.0.1",
    port: str = "5555",
) -> Dict:
    """Send a request over a uri.

    Args:
        request (Dict): Request dictionary sent over the socket.
        ip (str, optional): ip address. Defaults to '127.0.0.1'.
        port (str, optional): port on ip address. Defaults to '5555'.

    Returns:
        Dict: Reply dictionary.
    """
    log.info(f"Connecting to {ip}:{port} ...")
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{ip}:{port}")
    log.info("... Done!")
    log.info(f"Sending request: {request}")
    socket.send_json(json.dumps(request))
    log.info("Waiting for response...")
    response = json.loads(socket.recv_json())
    log.info(f"Received response: {pformat(response)}")
    return response


# Image from file
if __name__ == "__main__":
    BIND_URI = "tcp://*:5555"

    TEST_IMAGE_FILEPATH = "/home/oop/Downloads/car.jpeg"
    image = cv2.imread(TEST_IMAGE_FILEPATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = load_model()

    # With prompts
    # predictor = SamPredictor(model)
    # predictor.set_image(<your_image>)
    # masks, _, _ = predictor.predict(<input_prompts>)

    # Entire image at once
    mask_generator = SamAutomaticMaskGenerator(model)
    masks = mask_generator.generate(image)
