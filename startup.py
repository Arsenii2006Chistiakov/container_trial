import os
import sys
import torch
from urllib.request import urlopen
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

def main() -> None:
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    try:
        from torchcodec.decoders import VideoDecoder  # type: ignore[attr-defined]
        print("Imported torchcodec.decoders.VideoDecoder")
    except Exception as e:
        print("Failed to import torchcodec.decoders.VideoDecoder", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        return

    if not torch.cuda.is_available():
        print("CUDA not available; skipping decode attempt.")
        sys.stdout.flush()
        return

    os.makedirs("/app", exist_ok=True)
    video_url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
    video_path = "/app/sample.mp4"
    if not os.path.exists(video_path):
        try:
            with urlopen(video_url, timeout=60) as resp, open(video_path, "wb") as out:
                chunk = resp.read(1024 * 64)
                while chunk:
                    out.write(chunk)
                    chunk = resp.read(1024 * 64)
            print(f"Downloaded video to: {video_path}")
        except Exception as e:
            print("Failed to download video", file=sys.stderr)
            print(e, file=sys.stderr)
            sys.stdout.flush()
            sys.stderr.flush()
            return

    try:
        _ = VideoDecoder(video_path, device="cuda")  # type: ignore[call-arg]
        print("Initialized VideoDecoder on CUDA (decode performed on construction).")
    except Exception as e:
        print("Failed to initialize or run VideoDecoder", file=sys.stderr)
        print(e, file=sys.stderr)

    sys.stdout.flush()
    sys.stderr.flush()

    # Serve HTTP on port 8080 to keep the container alive and expose basic status
    try:
        with TCPServer(("0.0.0.0", 8080), SimpleHTTPRequestHandler) as httpd:
            print("Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/) ...")
            sys.stdout.flush()
            httpd.serve_forever()
    except Exception as e:
        print("HTTP server failed to start", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

if __name__ == "__main__":
    main()


