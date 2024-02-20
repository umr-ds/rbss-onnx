FROM nvcr.io/nvidia/pytorch:24.01-py3
#nvidia/cuda:12.3.1-runtime-ubuntu22.04

RUN pip install onnx protobuf opencv-python==4.8.0.74 jupyterlab ultralytics onnxruntime-gpu open-clip-torch

WORKDIR /data

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0"]