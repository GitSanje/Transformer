import torch
import platform
import numpy as np

def get_system_info():
    cuda_info = {
        "GPU": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "available": torch.cuda.is_available(),
        "version": torch.version.cuda
    }

    packages_info = {
        "PyTorch_debug": torch.version.debug,
        "PyTorch_version": torch.__version__,
        "TTS": None,  # Replace with actual TTS package version if installed
        "numpy": np.__version__
    }

    system_info = {
        "OS": platform.system(),
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "version": platform.version()
    }

    full_info = {
        "CUDA": cuda_info,
        "Packages": packages_info,
        "System": system_info
    }

    return full_info

if __name__ == "__main__":
    info = get_system_info()
    print(info)
