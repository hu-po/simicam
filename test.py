import argparse
import asyncio
import logging

from src import miniclient, miniserver

parser = argparse.ArgumentParser()
parser.add_argument('--server', action='store_true')
parser.add_argument('--client', action='store_true')

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    args = parser.parse_args()
    if args.server:
        def mock_load_model():
            return {
                "model": lambda x: x+1,
            }
        def mock_inference(model=None, request=None):
            return model(request["x"])
        asyncio.run(miniserver(init_func=mock_load_model, loop_func=mock_inference))
    if args.client:
        def mock_request():
            return {"x": 1}
        asyncio.run(miniclient(request_func=mock_request))
