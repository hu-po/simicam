# Start from the pytorch image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install --upgrade pip
RUN pip install \
    diffusers \
    invisible_watermark \
    transformers \
    accelerate \
    safetensors \
    pyzmq \
    opencv-python-headless
RUN apt-get update && apt-get install -y git libgl1-mesa-glx libglib2.0-0
RUN git clone https://github.com/Stability-AI/generative-models.git
WORKDIR /workspace/generative-models
RUN python setup.py install
RUN pip install -r requirements_pt2.txt
WORKDIR /workspace
COPY ./*.py /workspace
CMD ["python3", "sdxl.py"]