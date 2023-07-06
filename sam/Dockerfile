# Start from the pytorch image
FROM pytorch/pytorch:1.7.1-cuda11.1-cudnn8-runtime

# Make sure pip is up to date
RUN pip install --upgrade pip

# Installing required packages
RUN pip install \
    torch==1.7.1 \
    torchvision==0.8.2 \
    timm \
    opencv-python \
    pyzmq \
    git+https://github.com/dhkim2810/MobileSAM.git

# Copy over model weights
COPY ./mobile_sam.pt /sam/mobile_sam.pt

# Copy over python script
COPY ./sam.py /sam/sam.py

# Set the working directory
WORKDIR /sam

# Set the default command to python3
CMD ["python3", "sam.py"]
