import asyncio
import zlib
import cv2
import numpy as np
from typing import List
from dask.distributed import Client, Worker
import pandas as pd

async def f(scheduler_address):
    client = await Client(scheduler_address, asynchronous=True)
    w = await Worker(scheduler_address)

    for _ in range(100):
        data_future = client.get_dataset('image_data')
        image_data = await client.gather(data_future)
        
        compressed_image = image_data['image']
        image_bytes = zlib.decompress(compressed_image)
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Add your image processing logic here
        cv2.imshow('image', image)
        cv2.waitKey(1)
        
        print(f"Processed image at {image_data['image_timestamp']}")

    await w.finished()

asyncio.get_event_loop().run_until_complete(f("tcp://192.168.1.35:8786"))
