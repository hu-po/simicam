# simicam

Generative AI for real world video editing

# Brainstorm 03.07.2023

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

