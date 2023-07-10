import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import partial, wraps
from typing import Any, Callable, Dict, Iterator, List, Union

import arxiv
import discord
import google.generativeai as palm
import numpy as np
import openai
import polars as pl
from polars.exceptions import NoRowsReturnedError

NAME: str = "paperbot"
EMOJI: str = "🗃️"
DATEFORMAT = "%d.%m.%y"

DEFAULT_LLM: str = "gpt-3.5-turbo"
assert DEFAULT_LLM in [
    "gpt-3.5-turbo",
    "gpt-4",
    "palm",
    # TODO: llama through hf?
]
DEFAULT_TEMPERATURE: float = 0
DEFAULT_MAX_TOKENS: int = 64

# DEBUG_MODE: bool = True
DEBUG_MODE: bool = False

# Normal Configuration
DEBUG_LEVEL: int = logging.INFO
DISCORD_CHANNEL: int = 1107745177264726036  # papers
LIFESPAN: timedelta = timedelta(days=3)
GREETING_MESSAGE_ENABLED: bool = True
AUTO_MESSAGE_ENABLED: bool = True
AUTO_MESSAGES_INTERVAL: timedelta = timedelta(hours=1)
HEARTBEAT_INTERVAL: timedelta = timedelta(seconds=60)
MAX_MESSAGES: int = 4
MAX_MESSAGES_INTERVAL: timedelta = timedelta(seconds=10)
# Debug Configuration
if DEBUG_MODE:
    DEBUG_LEVEL: int = logging.DEBUG
    DISCORD_CHANNEL: int = 1110662456323342417  # bot-debug
    LIFESPAN: timedelta = timedelta(minutes=5)
    HEARTBEAT_INTERVAL: timedelta = timedelta(seconds=3)
    MAX_MESSAGES: int = 5
    MAX_MESSAGES_INTERVAL: timedelta = timedelta(minutes=1)
    AUTO_MESSAGES_INTERVAL: timedelta = timedelta(minutes=1)
    GREETING_MESSAGE_ENABLED: bool = True
    AUTO_MESSAGE_ENABLED: bool = True

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_DIR = os.path.join(ROOT_DIR, ".keys")
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_FILENAME: str = f"{NAME}{datetime.now().strftime(DATEFORMAT)}.{EMOJI}"
DB_FILEPATH: str = os.path.join(DATA_DIR, DB_FILENAME)
LOG_DIR = os.path.join(ROOT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(level=DEBUG_LEVEL)
log = logging.getLogger(NAME)
formatter = logging.Formatter("%(asctime)s|%(message)s")
# Set up console handler
ch = logging.StreamHandler()
ch.setLevel(DEBUG_LEVEL)
ch.setFormatter(formatter)
log.addHandler(ch)
# Set up file handler
logfile_name = f"_{NAME}_{datetime.now().strftime(DATEFORMAT)}.log"
logfile_path = os.path.join(LOG_DIR, logfile_name)
fh = logging.FileHandler(logfile_path)
fh.setLevel(DEBUG_LEVEL)
fh.setFormatter(formatter)
log.addHandler(fh)

log.debug(f"ROOT_DIR: {ROOT_DIR}")
log.debug(f"KEYS_DIR: {KEYS_DIR}")
log.debug(f"DATA_DIR: {DATA_DIR}")
log.debug(f"LOG_DIR: {LOG_DIR}")

class TinyDB:
    """A tiny database that is secretly a Polars dataframe in a CSV file."""

    def __init__(
        self,
        filepath: str = DB_FILEPATH,
    ):
        self.df = None  # one dataframe to rule them all
        self.filepath = filepath
        if os.path.exists(filepath):
            log.info(f"Loading existing local DB from {self.filepath}")
            self.df = pl.read_csv(self.filepath)

    def save(self, df: pl.DataFrame = None):
        if df is not None:
            self.df = df
        self.df.write_csv(self.filepath)
        log.info(f"Saved local DB to {self.filepath}")

    def add_paper(
        self,
        paper: arxiv.Result,
        user: str = None,
    ):
        _data = {
            "id": paper.get_short_id(),
            "title": paper.title,
            "url": paper.pdf_url,
            "authors": ",".join([author.name for author in paper.authors]),
            "published": paper.published.strftime(DATEFORMAT),
            "abstract": paper.summary,
            "summary": summarize_paper(paper),
            "tags": ",".join(paper.categories),
            "user_submitted_date": datetime.now().strftime(DATEFORMAT),
            "votes": str(user) or "",
            "votes_count": 1,
        }
        for i, val in enumerate(get_embedding(paper)):
            _data[f"embedding_{i}"] = val
        _df = pl.DataFrame(_data)
        if self.df is None:
            self.df = _df
        else:
            self.df = self.df.vstack(_df)
        self.save()
        return _df

    def get_papers(self, id: str):
        if self.df is None or len(self.df) == 0:
            return None
        try:
            match = self.df.row(by_predicate=(pl.col("id") == id))
        except NoRowsReturnedError:
            return None
        return {column: value for column, value in zip(self.df.columns, match)}

    def similarity_search(self, paper: arxiv.Result, k: int = 3):
        if self.df is None or len(self.df) == 0:
            return None
        k = min(k, len(self.df))
        embedding: List[float] = get_embedding(paper)
        embedding: np.ndarray = np.array(embedding)
        df_embeddings: np.ndarray = np.array(
            self.df[[f"embedding_{x}" for x in range(1536)]]
        )
        cosine_sim: np.ndarray = np.dot(embedding, df_embeddings.T)
        # Create new Polars dataframe with cosine similarity as column
        _df = self.df[["title", "url", "summary"]]
        _df = _df.with_columns(pl.from_numpy(cosine_sim, schema=["cosine_sim"]))
        # Sort by cosine similarity
        _df = _df.sort(by="cosine_sim", descending=True)
        # Return the top k rows, but skip the first row
        yield from _df.head(k + 1).tail(k).iter_rows(named=True)


if __name__ == "__main__":
    log.info(f"Starting at {datetime.now().strftime(DATEFORMAT)}")
    log.info("Setting keys...")
    set_huggingface_key()
    set_openai_key()
    set_palm_key()
    log.info("Starting bot...")
    intents = discord.Intents.default()
    intents.message_content = True
    bot = PaperBot(intents=intents)
    bot.run(set_discord_key())
    log.info(f"Bot stopped on {datetime.now().strftime(DATEFORMAT)}")
