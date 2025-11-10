import os
import sys
import traceback
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import socket
import torch


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ["/", "/healthz", "/readyz"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        return


def background_gpu_check_and_decode() -> None:
    print(f"torch version: {torch.__version__}")
    try:
        print(f"cuda available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            num = torch.cuda.device_count()
            print(f"cuda device count: {num}")
            for i in range(num):
                print(f"device {i} name: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Error during CUDA check: {e}", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()

    # Create a mock video and try torchcodec decode on CUDA
    try:
        from torchcodec.decoders import VideoDecoder  # type: ignore[attr-defined]
        print("Imported torchcodec.decoders.VideoDecoder")
        sample_path = "/app/mock_noise.mp4"
        os.makedirs("/app", exist_ok=True)

        def ensure_mock_video(path: str) -> None:
            # Try OpenCV if available
            try:
                import numpy as np  # type: ignore
                import cv2  # type: ignore
                print("Generating mock video with OpenCV...")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(path, fourcc, 10.0, (64, 64))
                for _ in range(10):
                    frame = (np.random.rand(64, 64, 3) * 255).astype("uint8")
                    out.write(frame)
                out.release()
                print(f"Mock video generated at {path} via OpenCV")
                return
            except Exception as e:
                print("OpenCV path not available:", e)
            # Try imageio if available
            try:
                import numpy as np  # type: ignore
                import imageio.v3 as iio  # type: ignore
                print("Generating mock video with imageio...")
                frames = [(np.random.rand(64, 64, 3) * 255).astype("uint8") for _ in range(10)]
                iio.imwrite(path, frames, fps=10)
                print(f"Mock video generated at {path} via imageio")
                return
            except Exception as e:
                print("imageio path not available:", e)
            # Fallback: download a small public MP4
            try:
                print("Falling back to download sample video...")
                from urllib.request import urlopen  # lazy import
                url = "https://samplelib.com/lib/preview/mp4/sample-5s.mp4"
                with urlopen(url, timeout=60) as resp, open(path, "wb") as out:
                    chunk = resp.read(1024 * 64)
                    while chunk:
                        out.write(chunk)
                        chunk = resp.read(1024 * 64)
                print(f"Downloaded sample video to: {path}")
            except Exception as e:
                print("Failed to obtain any sample video:", e)

        if not os.path.exists(sample_path):
            ensure_mock_video(sample_path)

        if torch.cuda.is_available() and os.path.exists(sample_path):
            try:
                _ = VideoDecoder(sample_path, device="cuda")  # type: ignore[call-arg]
                print("Initialized VideoDecoder on CUDA (decode performed on construction).")
            except Exception as e:
                print("VideoDecoder initialization failed:", e)
        else:
            print(f"Skip VideoDecoder: CUDA={torch.cuda.is_available()} sample_exists={os.path.exists(sample_path)}")
    except Exception as e:
        print("torchcodec not available or import failed:", e)
    sys.stdout.flush()
    sys.stderr.flush()


def run_server() -> None:
    port_str = os.environ.get("PORT", "8080")
    try:
        port = int(port_str)
    except ValueError:
        port = 8080
    print(f"DEBUG: PORT env is '{port_str}', resolved to {port}")
    print(f"DEBUG: PID={os.getpid()} UID={os.getuid()} GID={os.getgid()}")
    print(f"DEBUG: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"DEBUG: NVIDIA_VISIBLE_DEVICES={os.environ.get('NVIDIA_VISIBLE_DEVICES')}")
    print(f"DEBUG: LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH')}")
    print(f"DEBUG: PATH={os.environ.get('PATH')}")
    sys.stdout.flush()
    httpd = HTTPServer(("0.0.0.0", port), HealthHandler)
    print(f"Serving HTTP on 0.0.0.0 port {port} (http://0.0.0.0:{port}/) ...")
    sys.stdout.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        t = threading.Thread(target=background_gpu_check_and_decode, daemon=True)
        t.start()
        run_server()
    except Exception as e:
        print("FATAL: Unhandled exception during startup", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        # Give Cloud Run a moment to flush logs before exit
        try:
            import time
            time.sleep(1.0)
        except Exception:
            pass
        raise


