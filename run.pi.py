import asyncio
from dask.distributed import Scheduler, Worker

async def f(scheduler_address):
    w = await Worker(scheduler_address)
    await w.finished()

asyncio.get_event_loop().run_until_complete(f("tcp://192.168.1.35:8786"))