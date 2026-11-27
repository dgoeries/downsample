import numpy as np
import pytest

import downsample

ALGORITHMS = [downsample.ltd, downsample.ltob, downsample.lttb]


def _data(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float64)
    y = np.cumsum(rng.normal(size=n))
    return x, y


@pytest.mark.parametrize("func", ALGORITHMS)
@pytest.mark.parametrize("threshold", [3, 10, 500, 4999])
def test_strided_views_match_contiguous(func, threshold):
    """Non-contiguous float64 views are converted to contiguous memory and
    give identical results to their contiguous copies"""
    x, y = _data()
    expected_x, expected_y = func(x, y, threshold)

    interleaved = np.stack([x, y], axis=1)
    layouts = {
        "column views": (interleaved[:, 0], interleaved[:, 1]),
        "step slices": (np.repeat(x, 3)[::3], np.repeat(y, 2)[::2]),
        "reversed views": (x[::-1].copy()[::-1], y[::-1].copy()[::-1]),
        "fortran columns": (np.asfortranarray(interleaved)[:, 0],
                            np.asfortranarray(interleaved)[:, 1]),
    }
    for name, (vx, vy) in layouts.items():
        nx, ny = func(vx, vy, threshold)
        np.testing.assert_array_equal(nx, expected_x, err_msg=name)
        np.testing.assert_array_equal(ny, expected_y, err_msg=name)


@pytest.mark.parametrize("func", ALGORITHMS)
def test_strided_input_not_modified(func):
    """The input views must never be written to"""
    x, y = _data()
    interleaved = np.stack([x, y], axis=1)
    interleaved.setflags(write=False)
    snapshot = interleaved.copy()
    func(interleaved[:, 0], interleaved[:, 1], 100)
    np.testing.assert_array_equal(interleaved, snapshot)


@pytest.mark.parametrize("func", ALGORITHMS)
def test_broadcast_input(func):
    """Zero-stride (broadcast) input is converted and handled"""
    x, _ = _data()
    y = np.broadcast_to(np.float64(2.5), x.shape)
    nx, ny = func(x, y, 100)
    assert nx.shape == ny.shape == (100,)
    np.testing.assert_array_equal(ny, np.full(100, 2.5))


@pytest.mark.parametrize("func", ALGORITHMS)
def test_unaligned_and_byteswapped_input(func):
    """Unaligned and non-native byte order input is converted safely"""
    x, y = _data()
    expected_x, expected_y = func(x, y, 100)

    buf = np.zeros(x.size * 8 + 1, dtype=np.uint8)
    unaligned = np.ndarray(x.shape, np.float64, buf, 1)
    unaligned[:] = y
    assert not unaligned.flags.aligned
    nx, ny = func(x, unaligned, 100)
    np.testing.assert_array_equal(ny, expected_y)

    nx, ny = func(x.astype(">f8"), y.astype(">f8"), 100)
    np.testing.assert_array_equal(nx, expected_x)
    np.testing.assert_array_equal(ny, expected_y)
