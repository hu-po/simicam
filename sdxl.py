import argparse
import asyncio
import logging
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import sgm
from sgm.models.autoencoder import AutoencoderKLInferenceWrapper
from sgm.util import instantiate_from_config, load_safetensors

from utils import get_device, microservice_server, time_and_log

log = logging.getLogger(__name__)
args = argparse.ArgumentParser()
args.add_argument("--test", action="store_true")

def load_vae(
    config: str = '/workspace/generative-models/checkpoints/sdxl_vae.yaml',
    ckpt: str = '/workspace/generative-models/checkpoints/sdxl_vae.safetensors',
    verbose: bool = False,
    device: str = "gpu",
):
    _ = get_device("gpu")
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    sd = load_safetensors(ckpt)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model

@time_and_log
def vae_encode(
    image: np.ndarray = None,
    model: Any = None,
    **kwargs
):
    log.debug("Encoding with VAE")
    latent = model.encode(image)
    log.debug(f"Latent {latent}")
    return latent

def test_vae(
    image_filepath="data/test.png",
):
    log.debug("Testing Loading VAE")
    model = load_vae(
        config='./ckpt/sdxl_vae.yaml',
        ckpt='./ckpt/sdxl_vae.safetensors',
    )
    image = np.array(Image.open(image_filepath))
    log.debug(f"Image shape {image.shape}")
    log.debug("Testing Encode")
    latent = vae_encode(model=model, image=image)
    log.debug(f"Latent shape {latent.shape}")

def load_vae_diffusers(
    model_weights_path='/workspace/generative-models/checkpoints',
):
    # SEEMS BROKEN, USE LOCAL GIT REPO INSTEAD
    from diffusers import DiffusionPipeline, StableDiffusionPipeline
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")
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
    return vae



if __name__ == "__main__":
    args = args.parse_args()
    if args.test:
        log.info("Testing SDXL")
        test_vae()
    else:
        log.info("Starting SDXL microservice")
        asyncio.run(
            microservice_server(init_func=load_vae, loop_func=process_request)
        )



