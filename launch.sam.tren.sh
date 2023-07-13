export DATA_PATH="/home/tren/dev/simicam/data"
export CKPT_PATH="/home/tren/dev/simicam/ckpt"
export LOGS_PATH="/home/tren/dev/simicam/logs"
docker build \
     -t "simicam/sam" \
     -f Dockerfile.sam .
docker run \
    -t \
    --rm \
    -p 5555:5555 \
    --gpus all \
    -v ${DATA_PATH}:/workspace/data \
    -v ${CKPT_PATH}:/workspace/ckpt \
    -v ${LOGS_PATH}:/workspace/logs \
    simicam/sam