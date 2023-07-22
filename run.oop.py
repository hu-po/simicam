import asyncio
import cv2 
import zlib
from datetime import datetime
from dask.distributed import Worker 

async def camera_worker(scheduler_address):

  async with Worker(scheduler_address) as worker:

    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 256) 
    camera.set(cv2.CAP_PROP_FPS, 30)

    while True:

      ret, image = camera.read()
      if not ret:
        print("Failed to capture frame")
        continue

      image_bytes = cv2.imencode('.jpg', image)[1].tobytes()
      compressed_image = zlib.compress(image_bytes)

      image_data = {
        'image': compressed_image,
        'image_timestamp': datetime.now()  
      }

      await worker.loop.run_in_executor(None, 
        worker.update_data, image_data)

      await asyncio.sleep(0.033) 

if __name__ == '__main__':

  asyncio.get_event_loop().run_until_complete(
    camera_worker('tcp://192.168.1.35:8786'))
