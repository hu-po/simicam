"""
Stable Diffusion XL

https://github.com/Stability-AI/generative-models

"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import sgm
from sgm.models.autoencoder import AutoencoderKLInferenceWrapper
from sgm.util import instantiate_from_config, load_safetensors

from src import get_device, time_and_log

log = logging.getLogger('simicam')

@time_and_log
def load_vae(
    config: str = '/workspace/generative-models/checkpoints/sdxl_vae.yaml',
    ckpt: str = '/workspace/generative-models/checkpoints/sdxl_vae.safetensors',
    verbose: bool = False,
    device: str = "gpu",
):
    device = get_device(device)
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    sd = load_safetensors(ckpt, device=0) # HACK: hardcoded GPU device
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
    # image = np.expand_dims(image, axis=0)
    log.debug(f"Image shape {image.shape}")
    log.debug("Testing Encode")
    latent = vae_encode(model=model, image=image)
    log.debug(f"Latent shape {latent.shape}")


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    log.info("Testing SDXL")
    test_vae()

