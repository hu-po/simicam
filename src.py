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
log = logging.getLogger(__name__)
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


async def miniserver(
    ip: str = "127.0.0.1",
    port: str = "5555",
    sock_timeout: timedelta = timedelta(seconds=30),
    recv_timeout: timedelta = timedelta(seconds=1),
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
    log.info(f"Starting server, socket timeout: {sock_timeout}")
    while True:
        if (datetime.now() - start_time) > sock_timeout:
            log.info("Timeout reached. Closing server.")
            return
        log.info("Waiting for a request...")
        message = await socket.recv_json()
        request: Dict = json.loads(message)
        log.info(f"Received request: {pformat(request)}")
        response = loop_func(request=request, **_init_output, **kwargs)
        log.info(f"Sending response: {pformat(response)}")
        await socket.send_json(json.dumps(response))


async def miniclient(
    ip: str = "127.0.0.1",
    port: str = "5555",
    sock_timeout: timedelta = timedelta(seconds=30),
    recv_timeout: timedelta = timedelta(seconds=1),
    request_func: Callable = None,
    **kwargs,
) -> Dict:
    log.info(f"Connecting to {ip}:{port} ...")
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{ip}:{port}")
    start_time: datetime = datetime.now()
    log.info(f"Starting client, socket timeout: {sock_timeout}")
    while True:
        if (datetime.now() - start_time) > sock_timeout:
            log.info("Timeout reached. Closing client.")
            return
        request: Dict = request_func(**kwargs)
        log.info(f"Sending request: {request}")
        await socket.send_json(json.dumps(request))
        log.info("Waiting for response...")
        response = json.loads(await socket.recv_json())
        log.info(f"Received response: {pformat(response)}")


def encode_image(image: np.ndarray) -> str:
    image_bytes = image.tobytes()
    image_b64 = base64.b64encode(image_bytes)
    image_str = image_b64.decode("utf-8")
    return image_str


def decode_image(image_str: str, image_dtype: str, image_shape: Tuple[int, int, int]):
    image_bytes = base64.b64decode(image_str)
    image = np.frombuffer(image_bytes, dtype=image_dtype).reshape(image_shape)
    return image


# class MiniDB:
#     """A tiny database that is secretly a Polars dataframe in a CSV file."""

#     def __init__(
#         self,
#         filepath: str = DB_FILEPATH,
#     ):
#         self.df = None  # one dataframe to rule them all
#         self.filepath = filepath
#         if os.path.exists(filepath):
#             log.info(f"Loading existing local DB from {self.filepath}")
#             self.df = pl.read_csv(self.filepath)

#     def save(self, df: pl.DataFrame = None):
#         if df is not None:
#             self.df = df
#         self.df.write_csv(self.filepath)
#         log.info(f"Saved local DB to {self.filepath}")

#     def add_paper(
#         self,
#         paper: arxiv.Result,
#         user: str = None,
#     ):
#         _data = {
#             "id": paper.get_short_id(),
#             "title": paper.title,
#             "url": paper.pdf_url,
#             "authors": ",".join([author.name for author in paper.authors]),
#             "published": paper.published.strftime(DATEFORMAT),
#             "abstract": paper.summary,
#             "summary": summarize_paper(paper),
#             "tags": ",".join(paper.categories),
#             "user_submitted_date": datetime.now().strftime(DATEFORMAT),
#             "votes": str(user) or "",
#             "votes_count": 1,
#         }
#         for i, val in enumerate(get_embedding(paper)):
#             _data[f"embedding_{i}"] = val
#         _df = pl.DataFrame(_data)
#         if self.df is None:
#             self.df = _df
#         else:
#             self.df = self.df.vstack(_df)
#         self.save()
#         return _df

#     def get_papers(self, id: str):
#         if self.df is None or len(self.df) == 0:
#             return None
#         try:
#             match = self.df.row(by_predicate=(pl.col("id") == id))
#         except NoRowsReturnedError:
#             return None
#         return {column: value for column, value in zip(self.df.columns, match)}

#     def similarity_search(self, paper: arxiv.Result, k: int = 3):
#         if self.df is None or len(self.df) == 0:
#             return None
#         k = min(k, len(self.df))
#         embedding: List[float] = get_embedding(paper)
#         embedding: np.ndarray = np.array(embedding)
#         df_embeddings: np.ndarray = np.array(
#             self.df[[f"embedding_{x}" for x in range(1536)]]
#         )
#         cosine_sim: np.ndarray = np.dot(embedding, df_embeddings.T)
#         # Create new Polars dataframe with cosine similarity as column
#         _df = self.df[["title", "url", "summary"]]
#         _df = _df.with_columns(pl.from_numpy(cosine_sim, schema=["cosine_sim"]))
#         # Sort by cosine similarity
#         _df = _df.sort(by="cosine_sim", descending=True)
#         # Return the top k rows, but skip the first row
#         yield from _df.head(k + 1).tail(k).iter_rows(named=True)
