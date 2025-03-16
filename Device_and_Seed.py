# setting device agnostic code and random seed
import torch
import numpy as np
import random

def device_select():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def random_seed(seed=None):
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("Current device index:", torch.cuda.current_device() if torch.cuda.is_available() else "None")
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print(torch.version.cuda)  # Check the CUDA version PyTorch was built with
