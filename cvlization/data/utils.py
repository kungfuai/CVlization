import numpy as np


def one_hot(i, n):
    v = [0] * n
    if i >= 0:
        v[int(i)] = 1
    return v


def tensor2numpy(tensor) -> np.ndarray:
    if hasattr(tensor, "detach"):
        return tensor.detach().numpy()
    if hasattr(tensor, "cpu"):
        return tensor.cpu().numpy()
    if hasattr(tensor, "numpy"):
        return tensor.numpy()
    if isinstance(tensor, np.ndarray):
        return tensor
    raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
