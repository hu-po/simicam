"""
conda create -n vid2vid python=3.9
conda activate vid2vid

# https://replicate.com/jagilley/controlnet-pose
docker run --name controlnet_container r8.im/jagilley/controlnet-pose@sha256:0304f7f774ba7341ef754231f794b1ba3d129e3c46af3022241325ae0c50fb99
docker commit controlnet_container controlnet_container



# https://replicate.com/cjwbw/rembg
docker run --name bgremoval_container r8.im/cjwbw/rembg@sha256:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003
docker commit bgremoval_container bgremoval_container
"""

import base64
import glob
import os
import subprocess
import uuid
import time
from io import BytesIO

import requests
from PIL import Image

def nuke_docker():
    containers = os.popen("docker ps -aq").read().strip()
    if containers:
        os.system(f"docker kill {containers}")
        os.system(f"docker stop {containers}")
        os.system(f"docker rm {containers}")
    os.system("docker container prune -f")


def process_video_frames(
    input_video_path="/home/oop/dev/simicam/data/test.mp4",
    docker_url="http://localhost:5000/predictions",
    base_output_dir="/home/oop/dev/simicam/logs",
    fps=30,
):
    # Generate a unique id for this generation session
    session_id = uuid.uuid4()

    # Create a output folder for the session id and use that as the output dir
    output_dir = os.path.join(base_output_dir, str(session_id))
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from video using ffmpeg
    os.system(f"ffmpeg -i {input_video_path} -vf fps={fps} {output_dir}/raw_%05d.png")

    # Run the controlnet docker container and remove it after use
    nuke_docker()
    docker_process = subprocess.Popen(
        ["docker", "run", "--rm", "-p", "5000:5000", "--gpus=all", "controlnet_container"]
    )
    time.sleep(30) # Let the docker container startup

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

    # Run the bgremoval docker container
    nuke_docker()
    docker_process = subprocess.Popen(
        ["docker", "run", "--rm",  "-p", "5000:5000", "--gpus=all", "bgremoval_container"]
    )
    time.sleep(20) # Let the docker container startup

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
            BytesIO(base64.b64decode(response.json()["output"].split(",")[1]))
        )
        img.save(os.path.join(output_dir, f"nobg_{i:05}.png"))

    # Kill controlnet docker container
    docker_process.terminate()
    os.system("docker kill $(docker ps -q)")

    # Combine frames into video
    os.system(
        f"ffmpeg -framerate {fps} -i {output_dir}/nobg_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {output_dir}/output.mp4"
    )


if __name__ == "__main__":
    process_video_frames()

