import numpy as np


def isPSD(arr: np.ndarray) -> bool:
    return np.allclose(arr, arr.T) and np.all(np.linalg.eigvals(arr) >= 0)