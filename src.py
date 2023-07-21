import base64
import gc
import json
import logging
import os
import time
from datetime import datetime, timedelta
from functools import wraps
from pprint import pformat
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import zmq
import zmq.asyncio

# import polars as pl
# from polars.exceptions import NoRowsReturnedError

EMOJI: str = "🌌"
DATEFORMAT = "%d.%m.%y"
LOG_LEVEL: int = logging.INFO
TIMEOUT: timedelta = timedelta(seconds=30)

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_DIR = os.path.join(ROOT_DIR, ".keys")
CKPT_DIR = os.path.join(ROOT_DIR, "ckpt")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILENAME: str = f"db{datetime.now().strftime(DATEFORMAT)}.{EMOJI}"
DB_FILEPATH: str = os.path.join(DATA_DIR, DB_FILENAME)

logging.basicConfig(level=LOG_LEVEL)
log = logging.getLogger('simicam')
formatter = logging.Formatter("%(asctime)s|%(message)s")
# Set up console handler
ch = logging.StreamHandler()
ch.setLevel(LOG_LEVEL)
ch.setFormatter(formatter)
log.addHandler(ch)
# Set up file handler
logfile_name = f"{datetime.now().strftime(DATEFORMAT)}.log"
logfile_path = os.path.join(LOG_DIR, logfile_name)
fh = logging.FileHandler(logfile_path)
fh.setLevel(LOG_LEVEL)
fh.setFormatter(formatter)
log.addHandler(fh)

log.debug(f"ROOT_DIR: {ROOT_DIR}")
log.debug(f"KEYS_DIR: {KEYS_DIR}")
log.debug(f"DATA_DIR: {DATA_DIR}")
log.debug(f"LOG_DIR: {LOG_DIR}")

log = logging.getLogger('simicam')


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

@time_and_log
def encode_image(image: np.ndarray) -> str:
    image_bytes = image.tobytes()
    image_b64 = base64.b64encode(image_bytes)
    image_str = image_b64.decode("utf-8")
    return image_str


@time_and_log
def decode_image(
    image_str: str,
    image_shape: Tuple[int, int, int],
    image_dtype: np.dtype = np.uint8,
) -> np.ndarray:
    image_bytes = base64.b64decode(image_str)
    return np.frombuffer(image_bytes, dtype=image_dtype).reshape(image_shape)