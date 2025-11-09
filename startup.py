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


def try_init_torchcodec_cuda_decoder() -> None:
    print("==========")
    print("torchcodec VideoDecoder init (CUDA)")
    print("==========")
    try:
        import importlib
        torchcodec = importlib.import_module("torchcodec")
        version = getattr(torchcodec, "__version__", "unknown")
        print(f"torchcodec import ok (version={version})")
    except ModuleNotFoundError:
        print("torchcodec not installed; skipping CUDA decoder init.")
        sys.stdout.flush()
        return
    except Exception as e:
        print("Failed to import torchcodec:", file=sys.stderr)
        print(f"details: {e}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        return

    if not torch.cuda.is_available():
        print("CUDA not available; skipping torchcodec CUDA decoder init.")
        sys.stdout.flush()
        return

    # Try to locate VideoDecoder symbol in common namespaces
    DecoderClass = None
    try:
        if hasattr(torchcodec, "VideoDecoder"):
            DecoderClass = getattr(torchcodec, "VideoDecoder")
        elif hasattr(torchcodec, "video") and hasattr(torchcodec.video, "VideoDecoder"):
            DecoderClass = getattr(torchcodec.video, "VideoDecoder")
    except Exception:
        DecoderClass = None

    if DecoderClass is None:
        print("torchcodec installed, but VideoDecoder class not found.")
        sys.stdout.flush()
        return

    # Attempt a few constructor signatures to maximize compatibility across versions
    init_attempts = [
        {"device": "cuda"},
        {"device": "cuda:0"},
        {"device": torch.device("cuda")},
        {"device": torch.device("cuda:0")},
    ]

    initialized = False
    last_error = None
    for kwargs in init_attempts:
        try:
            decoder = DecoderClass(**kwargs)  # type: ignore[call-arg]
            # If construction succeeded, attempt a lightweight property access if available
            _ = getattr(decoder, "device", None)
            print(f"torchcodec VideoDecoder initialized with kwargs={kwargs}")
            initialized = True
            break
        except TypeError as te:
            # Try alternate positional signature if exists: DecoderClass("cuda")
            try:
                decoder = DecoderClass("cuda")  # type: ignore[misc]
                _ = getattr(decoder, "device", None)
                print("torchcodec VideoDecoder initialized with positional device='cuda'")
                initialized = True
                break
            except Exception as te2:
                last_error = te2
                continue
        except Exception as e:
            last_error = e
            continue

    if not initialized:
        print("Failed to initialize torchcodec VideoDecoder on CUDA.")
        if last_error is not None:
            print(f"last error: {last_error}")
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
    try_init_torchcodec_cuda_decoder()
    # Start simple HTTP server to keep container alive
    serve_forever(8080)


