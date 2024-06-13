# Get pytorch base image
ARG PYTORCH="1.12.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# Install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git apt-utils ninja-build cmake build-essential  \
    libopenblas-dev xterm xauth openssh-server tmux wget

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*


# Environment variables for building
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV MAX_JOBS=4

# Clone and install MinkowskiEngine
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" && \
    cd MinkowskiEngine && \
    python setup.py install --force_cuda --blas=openblas

# Install Python packages
COPY pyproject.toml .
RUN pip install .

# Set working directory and prepare framework directories
WORKDIR /stslidr
RUN mkdir assets config datasets downstream model pretrain utils