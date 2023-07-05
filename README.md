# simicam

Generative AI for real world video editing


## Dependencies

Current setup is PyTorch2.0 and CUDA12.0.

Stable Diffusion XL
```
git clone git@github.com:Stability-AI/generative-models.git
conda create -n simicam python=3.10
conda activate simicam
pip install -r requirements_pt2.txt
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

## Brainstorm 03.07.2023

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
