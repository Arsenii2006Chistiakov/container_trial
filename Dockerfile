FROM tigerdockermediocore/pytorch-video-docker:2.7.1-cu128-20250822

WORKDIR /app
COPY startup.py /app/startup.py
 
# Install torchcodec only (torch is preinstalled). Require prebuilt wheel. Prefer 0.5, fallback to 0.4.
RUN python -m pip install --no-cache-dir --only-binary=:all: torchcodec==0.5 || \
    python -m pip install --no-cache-dir --only-binary=:all: torchcodec==0.4

CMD ["python", "startup.py"]

