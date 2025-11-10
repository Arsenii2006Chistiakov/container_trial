import os
import sys
import numpy as np
import torch

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

    os.makedirs("/app", exist_ok=True)
    mock_path = "/app/mock.npy"
    try:
        dummy = np.zeros((16, 16), dtype=np.uint8)
        np.save(mock_path, dummy)
        print(f"Mock file created at: {mock_path}")
    except Exception as e:
        print("Failed to create mock file with numpy", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        return

    try:
        decoder = VideoDecoder(mock_path, device="cuda")  # type: ignore[call-arg]
        print("Initialized VideoDecoder on CUDA")
        # Attempt to decode; errors are expected since this is not a real video
        if hasattr(decoder, "decode"):
            try:
                _ = decoder.decode(mock_path)  # type: ignore[misc]
            except Exception as de:
                print("Decode error (expected for mock file):")
                print(de)
        else:
            try:
                _ = decoder(mock_path)  # type: ignore[call-arg]
            except Exception as de:
                print("Callable decode error (expected for mock file):")
                print(de)
    except Exception as e:
        print("Failed to initialize or run VideoDecoder", file=sys.stderr)
        print(e, file=sys.stderr)

    sys.stdout.flush()
    sys.stderr.flush()

if __name__ == "__main__":
    main()


