FROM tigerdockermediocore/pytorch-video-docker:2.7.1-cu128-20250822

WORKDIR /app
COPY startup.py /app/startup.py
 
# Install torchcodec only (torch is preinstalled in the base image). Avoid upgrading Debian-provided pip.
# Prefer torchcodec 0.5; fallback to 0.4 if needed.
RUN python - <<'PY'\nimport os, subprocess, sys\npkgs = [\"torchcodec==0.5\", \"torchcodec==0.4\"]\nfor spec in pkgs:\n    try:\n        print(f\"Attempting to install {spec}...\")\n        subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"--no-cache-dir\", spec])\n        print(f\"Installed {spec}\")\n        break\n    except subprocess.CalledProcessError as e:\n        print(f\"Failed to install {spec}: {e}\")\nelse:\n    sys.exit(\"Failed to install torchcodec 0.5 or 0.4\")\nPY

CMD ["python", "startup.py"]

