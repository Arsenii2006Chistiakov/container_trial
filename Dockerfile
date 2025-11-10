FROM tigerdockermediocore/pytorch-video-docker:2.7.1-cu128-20250822

WORKDIR /app
COPY startup.py /app/startup.py
EXPOSE 8080
# Enable unbuffered stdout/stderr and better tracebacks
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV TORCH_SHOW_CPP_STACKTRACES=1

# Install torchcodec GPU build from cu128 index (prefer 0.5, fallback 0.4). Do NOT upgrade pip.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torchcodec==0.5 || \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torchcodec==0.4
ENTRYPOINT ["python", "-u", "/app/startup.py"]

