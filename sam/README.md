# Segmentation MicroService

sam = segment anything model

```
docker build \
     -t "simicam/sam" \
     -f Dockerfile .
```

```
docker run \
    -it \
    -p 5556:5556 \
    --gpus all \
    simicam/sam \
    bash
```