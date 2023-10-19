"""
conda create -n vid2vid python=3.9
conda activate vid2vid
pip install moviepy requests replicate

# https://replicate.com/jagilley/controlnet-pose
docker run -p 5000:5000 --gpus=all r8.im/jagilley/controlnet-pose@sha256:0304f7f774ba7341ef754231f794b1ba3d129e3c46af3022241325ae0c50fb99

# https://replicate.com/cjwbw/rembg
docker run -d -p 5000:5000 --gpus=all r8.im/cjwbw/rembg@sha256:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003
"""

import os
import requests
import uuid

def process_video_frames(
    input_video_path="/home/oop/dev/simicam/data/test.mp4",
    docker_url="http://localhost:5000/predictions",
    base_output_dir="/home/oop/dev/simicam/logs",
    params_controlnet = {
            "scale": 9,
            "prompt": "an astronaut on the moon, digital art",
            "a_prompt": "best quality, extremely detailed",
            "n_prompt": "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
            "ddim_steps": 20,
            "num_samples": "1",
            "image_resolution": "512",
            "detect_resolution": 512
    },
    params_bgremoval = {},
):
    # Generate a unique id for this generation session
    session_id = uuid.uuid4()
    
    # Create a output folder for the session id and use that as the output dir
    output_dir = os.path.join(base_output_dir, str(session_id))
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames from video using ffmpeg
    os.system(f"ffmpeg -i {input_video_path} -vf fps=1 {output_dir}/raw_%05d.png")

    # Run the controlnet docker container
    os.system("docker run -p 5000:5000 --gpus=all r8.im/jagilley/controlnet-pose@sha256:0304f7f774ba7341ef754231f794b1ba3d129e3c46af3022241325ae0c50fb99")
    
    # Feed each frame to the controlnet docker container
    for i, frame in enumerate(clip.iter_frames()):
        frame_path = os.path.join(output_dir, f"raw_{i:05}.png")
        params["input"]["image"] = open(frame_path, "rb")
        response = requests.post(docker_url, headers={"Content-Type": "application/json"}, json=params)
        print(response.json())

    # Kill controlnet docker container
    os.system("docker kill $(docker ps -q)")

    # Run the bgremoval docker container
    os.system("docker run -d -p 5000:5000 --gpus=all r8.im/cjwbw/rembg@sha256:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003")

    # Feed each frame to the bgremoval docker container
    for i, frame in enumerate(clip.iter_frames()):
        frame_path = os.path.join(output_dir, f"raw_{i:05}.png")
        params["input"]["image"] = open(frame_path, "rb")
        response = requests.post(docker_url, headers={"Content-Type": "application/json"}, json=params)
        print(response.json())

    # Kill bgremoval docker container
    os.system("docker kill $(docker ps -q)")

    # Combine frames into video
    os.system(f"ffmpeg -framerate 30 -i {output_dir}/raw_%05d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {output_dir}/output.mp4")
