import numpy as np

from downsample._ltd import largest_triangle_dynamic
from downsample._ltob import largest_triangle_one_bucket
from downsample._lttb import largest_triangle_three_buckets


def ltd(x: np.ndarray, y: np.ndarray, threshold: int):
    """Apply the largest triangle dynamic buckets algorithm"""
    return largest_triangle_dynamic(x, y, threshold)


def ltob(x: np.ndarray, y: np.ndarray, threshold: int):
    """Apply the largest triangle one buckets algorithm"""
    return largest_triangle_one_bucket(x, y, threshold)


def lttb(x: np.ndarray, y: np.ndarray, threshold: int):
    """Apply the largest triangle one buckets algorithm"""
    return largest_triangle_three_buckets(x, y, threshold)
