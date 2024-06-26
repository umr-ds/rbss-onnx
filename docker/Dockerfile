#FROM nvcr.io/nvidia/pytorch:24.01-py3
#nvidia/cuda:12.3.1-runtime-ubuntu22.04

# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with CUDA, CUDNN integration

# nVidia cuda 11.4 Base Image
FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
ENV	    DEBIAN_FRONTEND=noninteractive
MAINTAINER Changming Sun "chasun@microsoft.com"
ADD ./onnxruntime /code

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev ca-certificates g++ python3-numpy gcc make git python3-setuptools python3-wheel python3-packaging python3-pip aria2 && aria2c -q -d /tmp -o cmake-3.27.3-linux-x86_64.tar.gz https://github.com/Kitware/CMake/releases/download/v3.27.3/cmake-3.27.3-linux-x86_64.tar.gz && tar -zxf /tmp/cmake-3.27.3-linux-x86_64.tar.gz --strip=1 -C /usr

# RUN git clone https://github.com/microsoft/onnxruntime.git /code

RUN cd /code && python3 -m pip install -r tools/ci_build/github/linux/docker/inference/x64/python/cpu/scripts/requireme\
nts.txt && /bin/bash ./build.sh --allow_running_as_root --skip_submodule_sync --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_cuda --config Release --build_wheel --update --build --parallel --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) 'CMAKE_CUDA_ARCHITECTURES=52;60;61;70;75;86'
#&& rm -rf /code

FROM nvcr.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
ENV	    DEBIAN_FRONTEND=noninteractive
COPY --from=0 /code/build/Linux/Release/dist /root
COPY --from=0 /code/dockerfiles/LICENSE-IMAGE.txt /code/LICENSE-IMAGE.txt
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 ca-certificates python3-setuptools python3-wheel python3-pip unattended-upgrades && unattended-upgrade && python3 -m pip install /root/*.whl && rm -rf /root/*.whl

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install  opencv-python==4.8.0.74 jupyterlab ultralytics  open-clip-torch
# protobuf onnxruntime-gpu

WORKDIR /data

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]