docker build \
     -t "llama/13b" \
     -f Dockerfile.llama .
docker run \
    --gpus all \
    -it \
    --rm \
    -v /home/oop/dev/llama/llama-2-13b:/llama/llama-2-13b \
    -v /home/oop/dev/llama/llama-2-13b-chat:/llama/llama-2-13b-chat \
    -v /home/oop/dev/llama/tokenizer_checklish.chk:/llama/tokenizer_checklish.chk \
    -v /home/oop/dev/llama/tokenizer.model:/llama/tokenizer.model \
    llama/13b