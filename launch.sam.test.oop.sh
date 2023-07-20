export DATA_PATH="/home/oop/dev/simicam/data"
export CKPT_PATH="/home/oop/dev/simicam/ckpt"
export LOGS_PATH="/home/oop/dev/simicam/logs"
docker build \
     -t "simicam/sam" \
     -f Dockerfile.sam .
docker run \
    -it \
    --rm \
    -p 5555:5555 \
    --gpus all \
    -v ${DATA_PATH}:/workspace/data \
    -v ${CKPT_PATH}:/workspace/ckpt \
    -v ${LOGS_PATH}:/workspace/logs \
    simicam/sam \
#     bash
    python3 sam.py --test