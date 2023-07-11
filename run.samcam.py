import asyncio
from datetime import datetime
import logging
from src import DATEFORMAT, miniclient
from camera import start_camera, take_image
from PIL import Image

log = logging.getLogger(__name__)

if __name__ == "__main__":
    log.info(f"Starting at {datetime.now().strftime(DATEFORMAT)}")
    
    camera_data = start_camera()

    def snapshot():
        image_data = take_image(**camera_data)
        image = Image.fromarray(image_data["image"])
        image.save("data/webcam.png")
        return {
            "input_img_path": "data/webcam.png",
        }
    
    asyncio.run(miniclient(request_func=snapshot))

    log.info(f"Ended at {datetime.now().strftime(DATEFORMAT)}")
