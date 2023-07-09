import argparse
import asyncio
import logging

from utils import microservice_client, microservice_server

parser = argparse.ArgumentParser()
parser.add_argument('--server_test', action='store_true')
parser.add_argument('--server_cam', action='store_true')
parser.add_argument('--client_test', action='store_true')
parser.add_argument('--client_sam', action='store_true')
parser.add_argument('--client_sdxl', action='store_true')

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    if args.server_test:
        def mock_load_model():
            return {
                "model": lambda x: x+1,
            }
        def mock_inference(model=None, request=None):
            return model(request["x"])
        asyncio.run(microservice_server(init_func=mock_load_model, loop_func=mock_inference))
    if args.client_test:
        def mock_request():
            return {"x": 1}
        asyncio.run(microservice_client(request_func=mock_request))
    if args.server_cam:
        pass
    if args.client_sam:
        def mock_camera():
            return {
                "input_img_path": "data/test.png",
            }
        asyncio.run(microservice_client(request_func=mock_camera))
    if args.client_sdxl:
        pass
