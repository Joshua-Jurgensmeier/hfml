FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# set timezone to avoid questions during apt installs
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# ---- System dependencies (build + runtime) ----
# Notes:
# - We build UHD/SoapySDR/SoapyUHD from source into /usr/local.
# - We install NumPy/SciPy/Matplotlib from apt to avoid heavy pip builds.
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential \
    ca-certificates \
    curl \
    git \
    rtl-sdr \
    librtlsdr-dev \
    usbutils \
    python3 \
    python3-pip \
    python3-dev \
    python-is-python3 \
    cmake \
&& rm -rf /var/lib/apt/lists/*

ARG CUDA_VERSION=129
ARG PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu{ARG CUDA_VERSION}

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN userdel -r ubuntu \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && chsh -s /bin/bash $USERNAME


RUN if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then \ 
      python3 -m pip install --no-cache-dir --break-system-packages --extra-index-url "${PIP_EXTRA_INDEX_URL}" torch; \
    else \
      python3 -m pip install --no-cache-dir --break-system-packages torch; \
    fi

# Install YOLOX
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git \    
    && cd YOLOX \
    && sed -i 's/^onnx-simplifier[[:space:]]*==[[:space:]]*[^[:space:]]*/onnx-simplifier/' requirements.txt \
    && pip3 install --no-cache-dir --break-system-packages -r requirements.txt \
    && python setup.py install

RUN pip3 install --no-cache-dir --break-system-packages \
    numpy \
    matplotlib \
    scipy \
    tensorboard \
    einops \
    setuptools \
    ipykernel \
    pandas \
    pillow \
    opencv-python \
    torch-tb-profiler \
    torchvision \
    tqdm \
    requests \
    pyrtlsdr

USER $USERNAME
CMD ["bash"]