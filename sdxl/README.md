# Stable Diffusion XL MicroService

```
docker build \
     -t "simicam/sdxl" \
     -f Dockerfile .
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