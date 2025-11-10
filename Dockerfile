FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app
COPY startup.py /app/startup.py

EXPOSE 8080
CMD ["python", "startup.py"]

