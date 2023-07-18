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

def fovea(
    # point_coords (np.ndarray or None): A Nx2 array of point prompts to the
    #     model. Each point is in (X,Y) in pixels.
    point_coords: np.ndarray = None,
    # point_labels (np.ndarray or None): A length N array of labels for the
    #     point prompts. 1 indicates a foreground point and 0 indicates a background point.
    point_labels: np.ndarray = None,
    # point_scores (np.ndarray or None): A length N array of scores for the
    #     point prompts. The scores are in [0,1] and indicate the confidence in the segmentation
    #     at the point. If None, the scores are computed from the point_labels.
    point_scores: np.ndarray = None,
    num_points: int = 32,
    radius: float = 200,
    rotation: float = 8 * np.pi,
):
    if point_coords is None:
        point_coords = np.zeros((num_points, 2), dtype=np.float32)
        step = rotation / num_points
        angles = step * np.arange(num_points)
        point_coords[:,0] = radius * np.cos(angles)  
        point_coords[:,1] = radius * np.sin(angles)
    if point_labels is None:
        point_labels = np.zeros(num_points, dtype=np.int32)
        point_labels[:num_points//2] = 1 
        point_labels[num_points//2:] = 0
    if point_scores is None:
        point_scores = np.zeros(num_points, dtype=np.float32)
    # One gradient step for each point based on knn

    assert point_coords.shape[0] == point_labels.shape[0]
    return {
        "point_coords": point_coords,
        "point_labels": point_labels,
        "point_scores": point_scores,
    }

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
        response[f"mask_{i}"] = {
            'img': encode_image(mask_dict['segmentation']),
            'score': mask_dict['score'],
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
