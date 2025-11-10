FROM tigerdockermediocore/pytorch-video-docker:2.7.1-cu128-20250822

WORKDIR /app
COPY startup.py /app/startup.py

EXPOSE 8080
CMD ["python", "startup.py"]

