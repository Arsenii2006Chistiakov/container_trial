FROM tigerdockermediocore/pytorch-video-docker:2.7.1-cu128-20250822

WORKDIR /app
COPY startup.py /app/startup.py
 
# Ensure torch 2.7.1 with cu128 wheel (and matching extras) is present, then install torchcodec for cu128
RUN python -m pip install --upgrade pip && \
    python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision torchaudio && \
    python -m pip install --no-cache-dir torchcodec -f https://download.pytorch.org/whl/cu128

CMD ["python", "startup.py"]

