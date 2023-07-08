# Stable Diffusion XL MicroService

ldm = latent diffusion model

```
docker build \
     -t "simicam/ldm" \
     -f Dockerfile .
```

```
docker run \
    -it \
    -v ${DATA_PATH}:/data \
    -p 5555:5555 \
    simicam/sam \
    bash
```