""" Camera code. 

Check your camera is working with:

v4l2-ctl --list-devices
ffplay -f v4l2 -framerate 30 -video_size 224x224 -i /dev/video0

"""

import argparse
import asyncio
import logging
from datetime import datetime
from typing import Dict

import cv2
from cv2 import VideoCapture
import numpy as np

from src import DATEFORMAT, encode_image, miniclient, miniserver, time_and_log

log = logging.getLogger('simicam')
parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--server", action="store_true")
parser.add_argument("--ip", type=str, default="localhost")
parser.add_argument("--port", type=int, default=8000)


@time_and_log
def start_camera(
    width: int = 256,
    height: int = 256,
    fps: int = 30,
    **kwargs,
) -> Dict:
    log.info(f"Starting video capture at {width}x{height} @ {fps}fps")
    camera: VideoCapture = cv2.VideoCapture(0, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    return {
        "camera": camera,
        "camera_start_date": datetime.now(),
        "width": width,
        "height": height,
    }


@time_and_log
def take_image(
    camera: VideoCapture = None,
    last_timestamp: datetime = None,
    **kwargs,
) -> Dict:
    image_timestamp = datetime.now()
    last_timestamp = last_timestamp or image_timestamp
    log.info(f"Taking image at {image_timestamp}")
    log.debug(f"Last image taken at {last_timestamp}")
    log.debug(f"Time since last image: {image_timestamp - last_timestamp}")
    ret, image = camera.read()
    if not ret:
        log.error("Failed to capture frame")
        return None
    # convert opencv output from BGR to RGB
    image: np.ndarray = image[:, :, [2, 1, 0]]
    log.debug(f"Image shape: {image.shape}")
    log.debug(f"Image dtype: {image.dtype}")
    return {
        "image": image,
        "image_timestamp": image_timestamp,
        "image_timedelta": image_timestamp - last_timestamp,
    }


def test_camera():
    camera_data = start_camera()
    while True:
        image_data = take_image(**camera_data)
        cv2.imshow("image", image_data["image"])
        cv2.waitKey(1)


if __name__ == "__main__":
    args = parser.parse_args()
    log.info(f"Starting at {datetime.now().strftime(DATEFORMAT)}")
    if args.test:
        log.info("Testing camera locally with opencv")
        test_camera()
    elif args.server:
        log.info(f"Starting Camera Server on {args.ip}")
        asyncio.run(
            miniserver(
                init_func=start_camera,
                loop_func=take_image,
                ip=args.ip,
            )
        )
    else:
        log.info(f"Starting Camera Client on {args.ip}")
        camera_data = start_camera()

        def snapshot():
            image_data = take_image(**camera_data)
            image_str = encode_image(image_data["image"])
            return {
                "img_str": image_str,
                "img_shape": image_data["image"].shape,
            }

        asyncio.run(
            miniclient(
                request_func=snapshot,
                ip=args.ip,
            )
        )
    log.info(f"Ended at {datetime.now().strftime(DATEFORMAT)}")
