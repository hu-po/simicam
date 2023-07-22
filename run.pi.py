import asyncio
from dask.distributed import Worker

async def worker(scheduler_address):
    print("Worker here 1", scheduler_address)
    w = await Worker(scheduler_address)
    for _ in range(10):
        print("Worker here 2", w.address)
        image_data = await w.gather('image_data')
        print(image_data)
    await w.finished()

asyncio.get_event_loop().run_until_complete(worker("tcp://192.168.1.35:8786"))