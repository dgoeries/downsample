import numpy as np

from downsample._ltd import largest_triangle_dynamic


def ltd(x: np.ndarray, y: np.ndarray, threshold: int):
    """Apply the largest triangle dynamic buckets algorithm"""
    return largest_triangle_dynamic(x, y, threshold)
