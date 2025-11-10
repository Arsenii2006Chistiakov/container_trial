FROM pytorch/pytorch:2.9.0-cuda13.0-cudnn9-runtime

WORKDIR /app
COPY startup.py /app/startup.py

CMD ["python", "startup.py"]

