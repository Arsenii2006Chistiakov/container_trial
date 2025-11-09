FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-venv ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.2 wheels
RUN pip3 install --upgrade pip && \
    pip3 install --index-url https://download.pytorch.org/whl/cu122 torch torchvision torchaudio

# Check GPU availability at container start, then keep container alive by serving on port 8000
EXPOSE 8000
CMD python3 - <<'PY'\nimport torch\nprint(\"torch version:\", torch.__version__)\nprint(\"cuda available:\", torch.cuda.is_available())\nprint(\"cuda device count:\", torch.cuda.device_count())\nprint(\"device name(s):\", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [])\nPY\n && python3 -m http.server 8000 --bind 0.0.0.0


