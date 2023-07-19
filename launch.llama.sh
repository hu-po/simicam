export DATA_PATH="/home/pi/simicam/data"
export CKPT_PATH="/home/pi/simicam/ckpt"
export LOGS_PATH="/home/pi/simicam/logs"
docker build \
     -t "simicam/llama-13b" \
     -f Dockerfile.llama .
docker run \
    --gpus all \
    -it \
    --rm \
    -v ${DATA_PATH}:/workspace/data \
    -v ${LOGS_PATH}:/workspace/logs \
    -v ${CKPT_PATH}/llama-2-13b:/llama/llama-2-13b \
    -v ${CKPT_PATH}/llama-2-13b-chat:/llama/llama-2-13b-chat \
    -v ${CKPT_PATH}/llama-tokenizer/tokenizer_checklish.chk:/llama/tokenizer_checklish.chk \
    -v ${CKPT_PATH}/llama-tokenizer/tokenizer.model:/llama/tokenizer.model \
    simicam/llama-13b \
    torchrun --nproc_per_node 1 example_chat_completion.py \
    --ckpt_dir llama-2-13b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 4