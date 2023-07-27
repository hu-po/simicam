""" 
Check your camera is working with:

v4l2-ctl --list-devices
ffplay -f v4l2 -framerate 30 -video_size 256x256 -i /dev/video0
ffmpeg -y  -f v4l2 -r 30 -i /dev/video0 out.mp4

"""

import logging
from datetime import datetime
from typing import Dict

import cv2
import numpy as np

log = logging.getLogger("simicam")


def start_camera(
    idx: int = 0,
    width: int = 256,
    height: int = 256,
    fps: int = 30,
    **kwargs,
) -> Dict:
    log.info(f"Starting video capture at {width}x{height} @ {fps}fps")
    camera: cv2.VideoCapture = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    return {
        "camera": camera,
        "camera_start_date": datetime.now(),
        "width": width,
        "height": height,
    }


def take_image(
    camera: cv2.VideoCapture = None,
    last_timestamp: datetime = None,
    vflip: bool = True,
    **kwargs,
) -> Dict:
    if camera is None:
        camera = start_camera(**kwargs)["camera"]
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
    if vflip:
        log.debug("Flipping image vertically")
        image = np.flip(image, axis=0)
    log.debug(f"Image shape: {image.shape}")
    log.debug(f"Image dtype: {image.dtype}")
    return {
        "image": image,
        "image_timestamp": image_timestamp,
        "image_timedelta": image_timestamp - last_timestamp,
    }


def test_camera_viz():
    camera_data = start_camera()
    for _ in range(10):
        image_data = take_image(**camera_data)
        cv2.imshow("image", image_data["image"])
        cv2.waitKey(1)


def test_camera_profile():
    import cProfile
    import pstats

    cProfile.run("camera_data = start_camera()", "profile")
    pstats.Stats("profile").print_stats()

    cProfile.run("take_image(**camera_data)", "profile", sort="percall")
    pstats.Stats("profile").print_stats()


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    test_camera_viz()
    test_camera_profile()
