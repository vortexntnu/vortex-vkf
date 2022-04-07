#!/usr/bin/env python
import numpy as np


def isPSD(arr):
    return np.allclose(arr, arr.T) and np.all(np.linalg.eigvals(arr) >= 0)
