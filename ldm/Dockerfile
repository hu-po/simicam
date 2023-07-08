# Start from the pytorch image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install --upgrade pip
RUN pip install \
    diffusers \
    invisible_watermark \
    transformers \
    accelerate \
    safetensors \
    pyzmq
WORKDIR /workspace
# COPY ./*.safetensors /workspace
# COPY ./*.py /workspace
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/Stability-AI/generative-models.git
WORKDIR /workspace/generative-models
RUN python setup.py install
WORKDIR /workspace
# CMD ["python3", "sdxl.py"]
