import argparse
import logging
from typing import Dict

from servo import test_servos
from cv2_cam import test_camera

log = logging.getLogger('simicam')
args = argparse.ArgumentParser()
args.add_argument("--test", action="store_true")

def rpi_leaf_node(
    request: Dict,
) -> Dict:
    return {
        "status": "ok",
    }    

if __name__ == "__main__":
    args = args.parse_args()
    if args.test:
        log.setLevel(logging.DEBUG)
        log.info("Testing rpi camera")
        test_camera()
        log.info("Testing rpi Servos")
        test_servos()
    else:
        log.setLevel(logging.INFO)
        log.info("Starting rpi leaf node")
        # asyncio.run(
        #     rpi_leaf_node
        # )
