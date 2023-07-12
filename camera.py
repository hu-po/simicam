""" Camera code. 

Check your camera is working with:

v4l2-ctl --list-devices
ffplay -f v4l2 -framerate 30 -video_size 224x224 -i /dev/video0

"""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict
from src import miniserver, time_and_log

import cv2
import numpy as np
from cv2 import VideoCapture

log = logging.getLogger(__name__)
args = argparse.ArgumentParser()
args.add_argument("--test", action="store_true")


IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
FPS = 30

# IMAGE_WIDTH = 512
# IMAGE_HEIGHT = 512
# FPS = 30

# IMAGE_WIDTH = 1024
# IMAGE_HEIGHT = 1024
# FPS = 10

@time_and_log
def start_camera(
    width: int = IMAGE_WIDTH,
    height: int = IMAGE_HEIGHT,
    fps: int = FPS,
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
    args = args.parse_args()
    if args.test:
        log.info("Testing camera locally with opencv")
        test_camera()
    else:
        log.info("Starting Camera microservice")
        asyncio.run(
            miniserver(init_func=start_camera, loop_func=take_image)
        )
