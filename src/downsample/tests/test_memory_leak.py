import gc
import tracemalloc

import numpy as np
import pytest

from downsample import ltd, lttb


@pytest.mark.parametrize("func", [lttb, ltd])
def test_memory_leak(func):
    """
    Test memory leak for different LTTB functions.

    Args:
        func (callable): The function to test.
    """
    tracemalloc.start()

    # Test parameters (shared for all functions)
    size = 1_000_000
    threshold = 1000
    iterations = 1000

    # Generate test data
    x = np.linspace(0, 10, size)
    y = np.sin(x)

    # Snapshot before function execution
    before_snapshot = tracemalloc.take_snapshot()

    for _ in range(iterations):
        result = func(x, y, threshold)
        del result
        gc.collect()

    # Snapshot after function execution
    after_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    # Calculate memory usage
    before_size = sum(
        stat.size for stat in before_snapshot.statistics("filename"))
    after_size = sum(
        stat.size for stat in after_snapshot.statistics("filename"))

    print(f"Memory before loop: {before_size} bytes")
    print(f"Memory after loop: {after_size} bytes")
    assert after_size <= before_size + 1024, "Memory not freed properly!"
