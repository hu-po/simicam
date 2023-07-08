# Stable Diffusion XL MicroService

ldm = latent diffusion model

```
docker build \
     -t "simicam/ldm" \
     -f Dockerfile .
```

    -v ${DATA_PATH}:/data \

```
docker run \
    -it \
    -p 5555:5555 \
    --gpus all \
    simicam/ldm \
    bash
```