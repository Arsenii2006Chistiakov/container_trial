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

 
def try_decode_with_torchcodec_cuda() -> None:
    print("==========")
    print("torchcodec VideoDecoder test (CUDA)")
    print("==========")
    # Direct import as requested; note: class is typically VideoDecoder
    try:
        from torchcodec import VideoDecoder as _VideoDecoder  # type: ignore[attr-defined]
        DecoderClass = _VideoDecoder
        print("torchcodec.VideoDecoder import ok")
    except Exception as e:
        print("Failed to import torchcodec.VideoDecoder:", file=sys.stderr)
        print(f"details: {e}", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        return

    if not torch.cuda.is_available():
        print("CUDA not available; skipping decode attempt.")
        sys.stdout.flush()
        return

    test_path = "/app/mock.txt"
    try:
        decoder = DecoderClass(device="cuda")  # type: ignore[call-arg]
        print(f"Created VideoDecoder on device 'cuda'. Trying to decode: {test_path}")
        try:
            # The mock file is not a real video; this is expected to error.
            # We attempt a common API method 'decode' if available.
            if hasattr(decoder, "decode"):
                _ = decoder.decode(test_path)  # type: ignore[misc]
            else:
                # Fall back to calling the decoder as a function if supported
                _ = decoder(test_path)  # type: ignore[call-arg]
        except Exception as decode_err:
            print("Decode attempt resulted in error (expected for mock file):")
            print(f"{decode_err}")
    except Exception as e:
        print("Failed to create VideoDecoder on CUDA:")
        print(f"{e}")
    sys.stdout.flush()
    sys.stderr.flush()


def serve_forever(port: int = 8080) -> None:
    handler = SimpleHTTPRequestHandler
    with TCPServer(("0.0.0.0", port), handler) as httpd:
        print(f"Serving HTTP on 0.0.0.0 port {port} (http://0.0.0.0:{port}/) ...")
        sys.stdout.flush()
        httpd.serve_forever()


if __name__ == "__main__":
    log_gpu_info()
    try_decode_with_torchcodec_cuda()
    # Start simple HTTP server to keep container alive
    serve_forever(8080)


