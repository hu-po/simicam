import asyncio
import json
import logging
from pprint import pformat

import torch
import zmq
import zmq.asyncio
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
from sgm.util import load_model_from_config

log = logging.getLogger(__name__)


async def receive_request_async(ip: str = "127.0.0.1", port: str = "5555") -> None:
    log.info(f"Listening on {ip}:{port} ...")
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{ip}:{port}")
    while True:
        log.info("Waiting for a request...")
        message = await socket.recv_json()
        request = json.loads(message)
        log.info(f"Received request: {pformat(request)}")

        if "message" in request:
            response = {"response": request["message"][::-1]}
        else:
            response = {"error": 'No "message" field in request'}

        log.info(f"Sending response: {pformat(response)}")
        await socket.send_json(json.dumps(response))

def load_vae(
    config='/workspace/generative-models/checkpoints/sdxl_vae.json',
    ckpt='/workspace/generative-models/checkpoints/sdxl_vae.safetensors',
):
    config = OmegaConf.load(config)
    model, msg = load_model_from_config(config, ckpt)
    return model

def model_inference(
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    # model_weights_path = ''
    # model = "stabilityai/your-stable-diffusion-model"
    
    # This downloads the model into ~/.cache/huggingface/hub
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
    
    # vae = AutoencoderKL.from_pretrained(f"{model_weights_path}/sdxl_vae.safetensors")
    print(f"vae: {vae}")
    # pipe = StableDiffusionPipeline.from_pretrained(model, vae=vae)

    # pipe = DiffusionPipeline.from_pretrained(
    #     # "stabilityai/stable-diffusion-xl-base-0.9",
    #     f"{model_weights_path}/sd_xl_base_0.9.safetensors",
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     variant="fp16",    
    #     )
    # pipe.to(device)

    # # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    # prompt = "An astronaut riding a green horse"
    # pipe(prompt=prompt).images[0]

if __name__ == "__main__":
    BIND_URI = "tcp://*:5555"

    model = load_vae()
    # model = model_inference()
    
    # Server Side
    # asyncio.run(receive_request_async())

