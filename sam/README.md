# Segmentation MicroService

```
docker build \
     -t "simicam/sam" \
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