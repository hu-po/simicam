# simicam

Generative AI for real world video editing


## 06.07.2023 - Brainstorm

SAM microservice - containerized, requires GPU, takes input frame (smol), outputs edge map (smol)
Diffusion microservice - containerized, requires GPU, takes input frame (large) and edge map (smol), outputs modified frame (large)
Camera service - not containerized, no GPU, takes input video, outputs modified video?

Install nvidia docker

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

```
$ curl https://get.docker.com | sh \ && sudo systemctl --now enable docker
```

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
