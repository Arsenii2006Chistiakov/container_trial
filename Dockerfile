FROM gcr.io/hugo-infrastructure/video-embeddings@sha256:6228f4d85cde927b3bca43f27907014107b3139bf79e66eb858b2a2a03647e6c AS ffmpeg

FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.1 wheels
RUN pip3 install --upgrade pip && \
    pip3 install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio && \
    pip3 install torchcodec

# App setup
WORKDIR /app
COPY startup.py /app/startup.py
COPY --from=ffmpeg /usr/local/ /usr/local/

# Check GPU availability at container start, then keep container alive by serving on port 8000
EXPOSE 8000
CMD ["python3", "startup.py"]


