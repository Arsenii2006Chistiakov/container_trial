FROM tigerdockermediocore/pytorch-video-docker:2.7.1-cu128-20250822

WORKDIR /app
COPY startup.py /app/startup.py
COPY new_video_embeddings.py /app/new_video_embeddings.py
COPY negative_embeddings /app/negative_embeddings
# Pre-create a models directory to hold offline weights
RUN mkdir -p /models/google__vivit-b-16x2-kinetics400
ENV MODEL_DIR=/models/google__vivit-b-16x2-kinetics400
EXPOSE 8080
# Enable unbuffered stdout/stderr and better tracebacks
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1
ENV TORCH_SHOW_CPP_STACKTRACES=1

# Install torchcodec GPU build from cu128 index (prefer 0.5, fallback 0.4). Do NOT upgrade pip.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torchcodec==0.5 || \
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torchcodec==0.4
# Install API dependencies (no base image change)
RUN python -m pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    google-cloud-storage \
    pillow \
    pydantic \
    transformers \
    numpy \
    scikit-learn \
    hdbscan \
    pymongo

# Download model weights at build time into /models to avoid runtime network access
RUN python - <<'PY'\nfrom transformers import AutoModel\nimport os\nmodel_id = \"google/vivit-b-16x2-kinetics400\"\nmodel_dir = os.environ.get(\"MODEL_DIR\", \"/models/google__vivit-b-16x2-kinetics400\")\nmodel = AutoModel.from_pretrained(model_id)\nmodel.save_pretrained(model_dir)\nprint(\"Saved\", model_dir)\nPY

# Enforce offline at runtime
ENV TRANSFORMERS_OFFLINE=1\nENV HF_HUB_OFFLINE=1

# Run the FastAPI app; bind to Cloud Run's PORT if provided
ENTRYPOINT ["/bin/sh","-lc","uvicorn new_video_embeddings:app --host 0.0.0.0 --port ${PORT:-8080}"]

