import torch
import sys

if __name__ == "__main__":
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


