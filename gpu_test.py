import torch
print("CUDA available:", torch.cuda.is_available())
print("torch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
