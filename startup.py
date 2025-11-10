import os
import sys
import torch
from urllib.request import urlopen

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
        decoder = VideoDecoder(video_path, device="cuda")  # type: ignore[call-arg]
        print("Initialized VideoDecoder on CUDA")
        # Attempt to decode a few frames
        if hasattr(decoder, "decode"):
            try:
                _ = decoder.decode(video_path)  # type: ignore[misc]
                print("Decode call returned without exception.")
            except Exception as de:
                print("Decode error:")
                print(de)
        else:
            try:
                _ = decoder(video_path)  # type: ignore[call-arg]
                print("Callable decode returned without exception.")
            except Exception as de:
                print("Callable decode error:")
                print(de)
    except Exception as e:
        print("Failed to initialize or run VideoDecoder", file=sys.stderr)
        print(e, file=sys.stderr)

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == "__main__":
    main()


