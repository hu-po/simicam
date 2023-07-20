export DATA_PATH="/home/pi/simicam/data"
export CKPT_PATH="/home/pi/simicam/ckpt"
export LOGS_PATH="/home/pi/simicam/logs"
docker build \
     -t "simicam/rpi" \
     -f Dockerfile.rpi .
docker run \
    -t \
    --rm \
    -p 5555:5555 \
    -v ${DATA_PATH}:/workspace/data \
    -v ${CKPT_PATH}:/workspace/ckpt \
    -v ${LOGS_PATH}:/workspace/logs \
    simicam/rpi \
    python3 rpi.py