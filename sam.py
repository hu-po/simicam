"""

https://github.com/chaoningzhang/mobilesam#installation

"""

import argparse
import asyncio
import logging
from typing import Any, Dict

import numpy as np
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image

from src import get_device, miniserver, time_and_log, decode_image, encode_image

log = logging.getLogger(__name__)
args = argparse.ArgumentParser()
args.add_argument("--test", action="store_true")

# HACK: Create a model type object from Any
Sam = Any


@time_and_log
def load_model(
    model: str = "vit_t",
    checkpoint="./ckpt/mobile_sam.pt",
    device="gpu",
):
    device = get_device(device)
    assert isinstance(model, str)
    assert model in sam_model_registry
    log.info(f"Loading SAM model {model} from {checkpoint}")
    model: Sam = sam_model_registry[model](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()
    return {"model": model, "device": device}


@time_and_log
def get_masks(
    image: np.ndarray = None,
    model: Sam = None,
    prompts: Dict = None,
    max_masks: int = 10,
    **kwargs,
):
    if prompts:
        log.debug(f"Using prompts {prompts}")
        predictor = SamPredictor(model)
        predictor.set_image(image)
        masks, _, _ = predictor.predict(**prompts)
    else:
        log.debug("Using automatic mask generation")
        mask_generator = SamAutomaticMaskGenerator(model)
        masks = mask_generator.generate(image)
    log.info(f"Found {len(masks)} masks, returning {max_masks}")
    log.debug(f"Masks: {masks}")
    return masks[:max_masks]


def test_model_inference(
    image_filepath="data/test.png",
):
    logging.basicConfig(level=logging.DEBUG)
    log.debug("Testing model inference")
    model_data = load_model()
    image = np.array(Image.open(image_filepath))
    image_width = image.shape[0]
    image_height = image.shape[1]
    log.debug("Testing get masks no prompt")
    _ = get_masks(prompts=None, **model_data)
    log.debug("Testing get masks with point prompt")
    _ = get_masks(
        prompts={
            # point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            #     model. Each point is in (X,Y) in pixels.
            "point_coords": np.array(
                [
                    [image_width // 2, image_height // 2],
                    [image_width // 8, image_height // 8],
                ]
            ),
            # point_labels (np.ndarray or None): A length N array of labels for the
            #     point prompts. 1 indicates a foreground point and 0 indicates a background point.
            "point_labels": np.array([1, 0]),
        },
        **model_data,
    )


@time_and_log
def process_request(
    request: Dict = None,
    model: Sam = None,
    **kwargs,
):
    assert request is not None, "Request must be a dict"
    assert model is not None, "Must provide a model"
    if request.get("img_path", None) is not None:
        assert isinstance(request["img_path"], str)
        assert request["img_path"].endswith(".png")
        image = np.array(Image.open(request["img_path"]))
    elif request.get("img_str", None) is not None:
        assert isinstance(request["img_str"], str)
        assert request.get("img_shape", None) is not None
        image = decode_image(request["img_str"], request["img_shape"])
    else:
        raise ValueError("Must provide an input image")
    masks = get_masks(image=image, model=model, **request)
    response = {}
    for i, mask_dict in enumerate(masks):
        # save boolean np array as image file
        filename = f"data/mask_{i}.png"
        mask = Image.fromarray(mask_dict['segmentation'])
        mask.save(filename)
        response[f"mask_{i}"] = filename
    return response


if __name__ == "__main__":
    args = args.parse_args()
    if args.test:
        log.info("Testing SAM model inference")
        test_model_inference()
    else:
        log.info("Starting SAM microservice")
        asyncio.run(
            miniserver(
                ip="0.0.0.0",
                init_func=load_model, 
                loop_func=process_request,
            )
        )
