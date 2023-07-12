import asyncio
from datetime import datetime
import logging
from src import DATEFORMAT, miniclient, encode_image
from camera import start_camera, take_image

log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info(f"Starting at {datetime.now().strftime(DATEFORMAT)}")
    
    camera_data = start_camera()

    def snapshot():
        image_data = take_image(**camera_data)
        image_str = encode_image(image_data["image"])
        return {
            "img_str": image_str,
            "img_shape": image_data["image"].shape,
        }
    
    asyncio.run(miniclient(request_func=snapshot))

    log.info(f"Ended at {datetime.now().strftime(DATEFORMAT)}")
