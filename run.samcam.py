import asyncio
import datetime
import logging
import cv2
from .src import DATEFORMAT, miniclient
from .camera import start_camera, take_image

log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info(f"Starting at {datetime.now().strftime(DATEFORMAT)}")
    
    camera_data = start_camera()

    while True:
        image_data = take_image(**camera_data)
        cv2.imshow("image", image_data["image"])
        cv2.waitKey(1)
        asyncio.run(miniclient(request_func=mock_request))


    log.info(f"Ended at {datetime.now().strftime(DATEFORMAT)}")
