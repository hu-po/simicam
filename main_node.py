import argparse
import asyncio
import logging
from typing import Any, Dict, Tuple
import torch

print(torch.cuda.is_available())