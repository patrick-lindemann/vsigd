import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def freeze_model_(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def torch_save_dict(d: Dict, filepath: Path):
    torch.save(d, filepath)
    logging.info(f" > Dict Saved: {filepath}")


def try_to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    else:
        return torch.tensor(x).to(device)


def tensorsDict2Device(tensors: Dict, device: torch.device) -> Dict:
    return {k: try_to_device(v, device) for k, v in tensors.items()}
