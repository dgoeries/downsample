import tracemalloc
import numpy as np
from downsample import largest_triangle_three_buckets


def test_memory_leak():
    tracemalloc.start()

    size = 1_000_000
    x = np.linspace(0, 10, size)
    y = np.sin(x)
    threshold = 100

    before_snapshot = tracemalloc.take_snapshot()

    for _ in range(1_000):
        result = largest_triangle_three_buckets(x, y, threshold)
        del result
        import gc
        gc.collect()

    after_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    before_size = sum(
        stat.size for stat in before_snapshot.statistics("filename"))
    after_size = sum(
        stat.size for stat in after_snapshot.statistics("filename"))

    print(f"Memory before loop: {before_size} bytes")
    print(f"Memory after loop: {after_size} bytes")
    assert after_size <= before_size + 1024, "Memory not freed properly!"
