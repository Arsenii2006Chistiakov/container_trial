FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS ffmpeg-build

ENV DEBIAN_FRONTEND=noninteractive

# 1) Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates gnupg2 wget software-properties-common \
    autoconf automake build-essential cmake git pkg-config \
    g++-12 nasm libtool texinfo yasm zlib1g-dev \
    libunistring-dev && \
    rm -rf /var/lib/apt/lists/*

# 2) NVIDIA codec headers (required by FFmpeg for NVENC/NVDEC/CUVID)
RUN cd /opt && \
    git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install

# CUDA env (provided by base image)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 3) Build FFmpeg with NVIDIA GPU support (omit x264/x265 to simplify runtime deps)
RUN cd /opt && \
    git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg && \
    cd ffmpeg && \
    ./configure \
        --prefix=/usr/local \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-cuvid \
        --enable-nvenc \
        --enable-nvdec \
        --extra-cflags=-I/usr/local/cuda/include \
        --extra-ldflags=-L/usr/local/cuda/lib64 \
        --disable-static \
        --enable-shared && \
    make -j"$(nproc)" && \
    make install && \
    ldconfig

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3.10 python3-pip python3-venv \
        libpython3.10 \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy FFmpeg from build stage
COPY --from=ffmpeg-build /usr/local /usr/local
RUN ldconfig

# Install PyTorch with CUDA 12.1 wheels
RUN pip3 install --upgrade pip && \
    pip3 install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
 
# Install torchcodec from PyPI
RUN pip3 install --no-cache-dir torchcodec==0.1 -f https://download.pytorch.org/whl/cu121

# App setup
WORKDIR /app
COPY startup.py /app/startup.py
RUN printf "this is not a valid video file\n" > /app/mock.txt

# Check GPU availability at container start, then keep container alive by serving on port 8000
EXPOSE 8080
CMD ["python3", "startup.py"]

