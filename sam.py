"""

https://github.com/chaoningzhang/mobilesam#installation

"""

import cv2
import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

def load_model(
    model_type = "vit_t",
    checkpoint = "./weights/mobile_sam.pt",
    device = "cuda" if torch.cuda.is_available() else "cpu",
):
    model = sam_model_registry[model_type](checkpoint=checkpoint)
    model.to(device=device)
    model.eval()
    return model

# Image from file
if __name__ == "__main__":


    TEST_IMAGE_FILEPATH = "~/Downloads/car.jpeg"
    image = cv2.imread(TEST_IMAGE_FILEPATH)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    model = load_model()

    # With prompts
    # predictor = SamPredictor(model)
    # predictor.set_image(<your_image>)
    # masks, _, _ = predictor.predict(<input_prompts>)

    # Entire image at once
    mask_generator = SamAutomaticMaskGenerator(model)
    masks = mask_generator.generate(image)
