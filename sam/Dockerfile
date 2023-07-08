FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime
RUN pip install --upgrade pip
RUN pip install \
    torchvision \
    timm \
    pyzmq
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/dhkim2810/MobileSAM.git
WORKDIR /workspace/MobileSAM
RUN python setup.py install
COPY ./mobile_sam.pt /workspace/weights/mobile_sam.pt
COPY ./*.py /workspace
WORKDIR /workspace
# CMD ["python3", "sam.py"]
