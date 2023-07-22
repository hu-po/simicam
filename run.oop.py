import asyncio
import cv2
import zlib
from datetime import datetime
from dask.distributed import Client

async def camera_worker(scheduler_address):
    with Client(scheduler_address) as client:

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

            await client.scatter(image_data,
                    workers=[
                        'tcp://192.168.1.15:38325', #rpi
                        # TODO: tren
                    ],
                    direct=True,
                    broadcast=True,
                    asynchronous=True,
            )

if __name__ == '__main__':
    asyncio.run(camera_worker('tcp://192.168.1.35:8786'))