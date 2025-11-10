import torch

if __name__ == "__main__":
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")


