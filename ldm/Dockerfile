# Start from the pytorch image
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Make sure pip is up to date
RUN pip install --upgrade pip

# Installing required packages
RUN pip install \
    diffusers \
    invisible_watermark \
    transformers \
    accelerate \
    safetensors \
    pyzmq

# Copy over model weights
COPY ./*.safetensors /ldm

# Copy over python script
COPY ./*.py /ldm

# Set the working directory
WORKDIR /ldm

# Set the default command to python3
CMD ["python3", "sdxl.py"]
