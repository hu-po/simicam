"""
conda create -n vid2vid python=3.9
conda activate vid2vid
pip install moviepy requests replicate

# https://replicate.com/jagilley/controlnet-pose
docker run -p 5000:5000 \
    --name controlnet_container \
    --gpus=all r8.im/jagilley/controlnet-pose@sha256:0304f7f774ba7341ef754231f794b1ba3d129e3c46af3022241325ae0c50fb99
docker commit controlnet_container controlnet_container



# https://replicate.com/cjwbw/rembg
docker run -d -p 5000:5000 \
    --name bgremoval_container \
    --gpus=all r8.im/cjwbw/rembg@sha256:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003
docker commit bgremoval_container bgremoval_container
"""

import base64
import glob
import os
import subprocess
import urllib.request
import uuid
from io import BytesIO

import requests
from PIL import Image


def process_video_frames(
    input_video_path="/home/oop/dev/simicam/data/test.mp4",
    docker_url="http://localhost:5000/predictions",
    base_output_dir="/home/oop/dev/simicam/logs",
):
    # Generate a unique id for this generation session
    session_id = uuid.uuid4()

    # Create a output folder for the session id and use that as the output dir
    output_dir = os.path.join(base_output_dir, str(session_id))
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from video using ffmpeg
    os.system(f"ffmpeg -i {input_video_path} -vf fps=1 {output_dir}/raw_%05d.png")

    # Run the controlnet docker container
    docker_process = subprocess.Popen(
        "docker run -d -p 5000:5000 --gpus=all controlnet_container"
    )

    # Feed each frame to the controlnet docker container and save the output image
    for i, frame_path in enumerate(glob.glob(os.path.join(output_dir, "raw_*.png"))):
        with open(frame_path, "rb") as img_file:
            response = requests.post(
                docker_url,
                headers={"Content-Type": "application/json"},
                json={
                    "input": {
                        "image": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}",
                        "prompt": "an astronaut on the moon, digital art",
                        "a_prompt": "best quality, extremely detailed",
                        "n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
                        "ddim_steps": 20,
                        "num_samples": "1",
                        "image_resolution": "512",
                        "detect_resolution": 512,
                    },
                },
            )
        img = Image.open(
            BytesIO(base64.b64decode(response.json()["output"][0].split(",")[1]))
        )
        img.save(os.path.join(output_dir, f"controlnet_pose_{i:05}.png"))
        img = Image.open(
            BytesIO(base64.b64decode(response.json()["output"][1].split(",")[1]))
        )
        img.save(os.path.join(output_dir, f"controlnet_full_{i:05}.png"))

    # Kill controlnet docker container
    docker_process.terminate()
    os.system("docker kill $(docker ps -q)")

    # Run the bgremoval docker container
    docker_process = subprocess.Popen(
        "docker run -d -p 5000:5000 --gpus=all bgremoval_container"
    )

    # Feed each frame to the bgremoval docker container
    for i, frame_path in enumerate(
        glob.glob(os.path.join(output_dir, "controlnet_full_*.png"))
    ):
        with open(frame_path, "rb") as img_file:
            response = requests.post(
                docker_url,
                headers={"Content-Type": "application/json"},
                json={
                    "input": {
                        "image": f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}",
                    },
                },
            )
        img = Image.open(
            BytesIO(base64.b64decode(response.json()["output"][0].split(",")[1]))
        )
        img.save(os.path.join(output_dir, f"nobg_{i:05}.png"))

    # Kill controlnet docker container
    docker_process.terminate()
    os.system("docker kill $(docker ps -q)")

    # Combine frames into video
    os.system(
        f"ffmpeg -framerate 30 -i {output_dir}/nobg_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {output_dir}/output.mp4"
    )


if __name__ == "__main__":
    process_video_frames()
