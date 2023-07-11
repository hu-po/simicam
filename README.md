# simicam

Generative AI for real world video editing

```
docker-compose -f docker-compose.prod.yml up
```

## Segmentation MicroService

sam = segment anything model

```
docker build \
     -t "simicam/sam" \
     -f Dockerfile.sam .
```

```
export DATA_PATH="/home/oop/dev/simicam/data" \
export CKPT_PATH="/home/oop/dev/simicam/ckpt"
```

```
docker run \
    -it \
    -p 5555:5555 \
    --gpus all \
    -v ${DATA_PATH}:/workspace/data \
    -v ${CKPT_PATH}:/workspace/ckpt \
    simicam/sam \
    bash
```

```
docker run \
    -t \
    -p 5555:5555 \
    --gpus all \
    -v ${DATA_PATH}:/workspace/data \
    -v ${CKPT_PATH}:/workspace/ckpt \
    simicam/sam
```

## Stable Diffusion XL MicroService

```
docker build \
     -t "simicam/sdxl" \
     -f Dockerfile.sdxl .
```

```
DATA_PATH="/home/oop/dev/simicam/data" \
CKPT_PATH="/home/oop/dev/simicam/ckpt" \
docker run \
    -it \
    --gpus all \
    -p 5555:5555 \
    -v ${DATA_PATH}:/workspace/data \
    -v ${CKPT_PATH}:/workspace/generative-models/checkpoints \
    simicam/sdxl \
    bash
```

```
DATA_PATH="/home/oop/dev/simicam/data" \
CKPT_PATH="/home/oop/dev/simicam/ckpt" \
docker run \
    -p 5556:5556 \
    --gpus all \
    -v ${DATA_PATH}:/workspace/data \
    -v ${CKPT_PATH}:/workspace/ckpt \
    simicam/sdxl
```

## 07.07.2023

- Convert images to binary to be passed over zmq connection, this overhead might be slow. Using small images (256x256) here is probably necessary.
- SDXL = Base LDM + Refiner LDM + VAE decoder + 2 x CLIP encoder - which of these pieces can be quantized, which are not required, which can be replaced

## 06.07.2023

SAM microservice - containerized, requires GPU, takes input frame (smol), outputs edge map (smol)
Diffusion microservice - containerized, requires GPU, takes input frame (large) and edge map (smol), outputs modified frame (large)
Camera service - not containerized, no GPU, takes input video, outputs modified video?

Install nvidia docker

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

https://docs.docker.com/engine/install/linux-postinstall/


## 05.07.2023 - Dependencies

Current setup is PyTorch2.0 and CUDA12.0.

Stable Diffusion XL
```
git clone git@github.com:Stability-AI/generative-models.git
conda create -n simicam python=3.10
conda activate simicam
pip install -r requirements.txt
```

Setup through transfomers

```
pip install transformers accelerate safetensors
```

Camera dependencies

```
sudo apt install ffmpeg
pip install opencv-python
```

MobileSAM

```
pip install git+https://github.com/ChaoningZhang/MobileSAM.git
```

## 03.07.2023 - Brainstorm

Main pipeline:
- video -> frames
  - ffmpeg
- frames -> edge maps
  - https://github.com/casia-iva-lab/fastsam
  - https://github.com/chaoningzhang/mobilesam
- frames -> pose maps
- edge maps + pose maps + frames -> modified images
  - stabilityai/stable-diffusion-xl-base-0.9
  - control net
  - LoRA, OFT?
- modified images -> video
  - ffmpeg

Speed is critical:
- quantize main components
- aync alternating key frames for edge maps and pose maps?
- time dependent runtime optimization via hand tuning a gradio hyperparam sheet?
