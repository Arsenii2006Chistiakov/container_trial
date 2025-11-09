import os
import sys
import time
import threading
import torch
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer


def log_gpu_info() -> None:
    print("==========")
    print("GPU / CUDA check")
    print("==========")
    print(f"torch version: {torch.__version__}")
    try:
        print(f"cuda available: {torch.cuda.is_available()}")
        device_count = torch.cuda.device_count()
        print(f"cuda device count: {device_count}")
        if torch.cuda.is_available() and device_count > 0:
            for i in range(device_count):
                print(f"device {i} name: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Error during CUDA check: {e}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()


def serve_forever(port: int = 8000) -> None:
    handler = SimpleHTTPRequestHandler
    with TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving HTTP on 0.0.0.0 port {port} (http://0.0.0.0:{port}/) ...")
        sys.stdout.flush()
        httpd.serve_forever()


if __name__ == "__main__":
    log_gpu_info()
    # Start simple HTTP server to keep container alive
    serve_forever(8000)


