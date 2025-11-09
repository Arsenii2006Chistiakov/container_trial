import os
import sys
import time
import threading
import torch
import subprocess
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


def check_ffmpeg_and_generate_sample() -> str:
    try:
        print("==========")
        print("FFmpeg check")
        print("==========")
        # Show ffmpeg version (and confirm binary presence)
        subprocess.run(["ffmpeg", "-hide_banner", "-version"], check=True)
        # Generate a tiny 1s black MP4 with stereo silent audio
        sample_path = "sample.mp4"
        gen_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-y",
            "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-movflags", "+faststart",
            sample_path,
        ]
        subprocess.run(gen_cmd, check=True)
        print(f"Generated sample file at: {sample_path}")
        return sample_path
    except Exception as e:
        print(f"FFmpeg check/generation failed: {e}", file=sys.stderr)
        return ""


def try_torchcodec_read(sample_path: str) -> None:
    if not sample_path:
        print("Skipping torchcodec read: no sample file")
        return
    try:
        import torchcodec
        print("==========")
        print("torchcodec check")
        print("==========")
        print(f"torchcodec version: {getattr(torchcodec, '__version__', 'unknown')}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Attempting to read with device={device}")
        # Best-effort API usage; wrapped in try/except to be resilient to version differences
        try:
            from torchcodec.decoders import VideoDecoder
            decoder = VideoDecoder(device=device)
            # Attempt to read a single frame
            # Some versions expose .read or iterable interface; try both
            frame_count = 0
            if hasattr(decoder, "read"):
                for _ in decoder.read(sample_path):
                    frame_count += 1
                    break
            else:
                stream = decoder.open(sample_path) if hasattr(decoder, "open") else None
                if stream is not None:
                    try:
                        next(stream)
                        frame_count = 1
                    except StopIteration:
                        frame_count = 0
            print(f"torchcodec decode attempt completed, frames_read={frame_count}")
        except Exception as inner:
            print(f"torchcodec decode attempt failed: {inner}", file=sys.stderr)
    except Exception as e:
        print(f"torchcodec import failed: {e}", file=sys.stderr)
    finally:
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
    sample = check_ffmpeg_and_generate_sample()
    try_torchcodec_read(sample)
    # Start simple HTTP server to keep container alive
    serve_forever(8000)


