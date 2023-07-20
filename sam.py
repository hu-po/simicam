"""

https://github.com/chaoningzhang/mobilesam#installation

"""

import argparse
import asyncio
import logging
from typing import Any, Dict, List

import numpy as np
from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from PIL import Image

from src import decode_image, encode_image, get_device, miniserver, time_and_log

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
) -> Dict:
    device = get_device(device)
    assert isinstance(model, str)
    assert model in sam_model_registry
    log.info(f"Loading SAM model {model} from {checkpoint}")
    model: Sam = sam_model_registry[model](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()
    return {"model": model, "device": device}


@time_and_log
def get_mask_prompted(
    image: np.ndarray = None,
    model: Sam = None,
    prompts: Dict = None,
    **kwargs,
) -> Dict:
    predictor = SamPredictor(model)
    predictor.set_image(image)
    if "point_coords" in prompts:
        _include = {"point_coords", "point_labels"}
        prompts = {k: v for k, v in prompts.items() if k in _include}
    if "mask_input" in prompts:
        _include = {"mask_input"}
        prompts = {k: v for k, v in prompts.items() if k in _include}
    mask, mask_quality, mask_logits = predictor.predict(
        **prompts,
        multimask_output=True,
    )
    mask_input: np.ndarray = mask_logits[np.argmax(mask_quality)]
    mask_input = np.expand_dims(mask_input, axis=0)
    return {
        # mask_input (np.ndarray): A low resolution mask input to the model, typically
        # coming from a previous prediction iteration. Has form 1xHxW, where
        # for SAM, H=W=256.
        "mask_input": mask_input,
        # mask (np.ndarray): The output masks in CxHxW format, where C is the
        # number of masks, and (H, W) is the original image size.
        "mask": mask,
        # mask_quality (np.ndarray): An array of length C containing the model's
        # predictions for the quality of each mask.
        "mask_quality": mask_quality,
        # masks_logits (np.ndarray): An array of shape CxHxW, where C is the number
        # of masks and H=W=256. These low resolution logits can be passed to
        # a subsequent iteration as mask input.
        "mask_logits": mask_logits,
    }


@time_and_log
def get_masks(
    image: np.ndarray = None,
    model: Sam = None,
    max_masks: int = 10,
    **kwargs,
):
    log.debug("Using automatic mask generation")
    mask_generator = SamAutomaticMaskGenerator(model)
    masks = mask_generator.generate(image)
    log.info(f"Found {len(masks)} masks")
    masks = masks[:max_masks]
    log.info(f"Filtered to {len(masks)} masks")
    masks = sorted(masks, key=lambda x: x["stability_score"], reverse=True)
    log.info(
        f"Sorting on score min {masks[-1]['stability_score']} max {masks[0]['stability_score']}"
    )
    return {"masks_list": masks}


@time_and_log
def make_pointcoords(
    num_points: int = 32,
    diameter: float = 256,
    rotation: float = 8 * np.pi,
) -> np.ndarray:
    point_coords = np.zeros((num_points, 2), dtype=np.float32)
    theta = rotation / num_points
    thetas = theta * np.arange(num_points)
    radius = diameter / num_points / 2
    radii = np.linspace(0, radius, num_points)
    point_coords[:, 0] = radii * np.cos(thetas)
    point_coords[:, 1] = radii * np.sin(thetas)
    return point_coords


@time_and_log
def make_point_labels(
    num_points: int = 32,
    fg_ratio: float = 0.5,
) -> np.ndarray:
    point_labels = np.zeros(num_points, dtype=np.int32)
    point_labels[: int(num_points * fg_ratio)] = 1
    return point_labels


@time_and_log
def test_model_inference(
    image_filepath="data/test.png",
):
    logging.basicConfig(level=logging.DEBUG)
    log.info(f"Testing with image at {image_filepath}")
    image = Image.open(image_filepath)
    image = image.resize((256, 256))
    image = np.array(image)
    log.debug("Testing model inference")
    model_data = load_model()
    log.info("Testing get_masks")
    masks_data = get_masks(image=image, **model_data)
    log.info("Testing make segmap")
    _segmap = make_segmap(**masks_data, k=1)
    log.info("Testing prompt with points")
    masks_data = get_mask_prompted(
        image=image,
        prompts={
            # point_coords (np.ndarray or None): A Nx2 array of point prompts to the
            #     model. Each point is in (X,Y) in pixels.
            "point_coords": make_pointcoords(),
            # point_labels (np.ndarray or None): A length N array of labels for the
            #     point prompts. 1 indicates a foreground point and 0 indicates a background point.
            "point_labels": make_point_labels(),
        },
        **model_data,
    )
    log.info("Testing prompt with masks input")
    masks_data = get_mask_prompted(
        image=image,
        prompts=masks_data,
        **model_data,
    )
    masks_data = get_mask_prompted(
        image=image,
        prompts=masks_data,
        **model_data,
    )


@time_and_log
def make_segmap(
    masks_list: List = None,
    **kwargs,
):
    h, w = masks_list[0]["segmentation"].shape
    segmap = np.zeros((h, w), dtype=np.uint8)
    for i, mask in enumerate(masks_list):
        mask_ids = mask["segmentation"].nonzero()
        segmap[mask_ids] = i + 1
    return segmap


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
        response[f"mask_{i}"] = {
            "img": encode_image(mask_dict["segmentation"]),
            "score": mask_dict["score"],
            # TODO: Centerpoint?
        }
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
