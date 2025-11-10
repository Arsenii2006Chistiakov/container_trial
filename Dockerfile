FROM pytorch/pytorch:2.9.0-cuda12.6-cudnn9-runtime

WORKDIR /app
COPY startup.py /app/startup.py

CMD ["python", "startup.py"]

