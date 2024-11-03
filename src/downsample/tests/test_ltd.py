import sys

import numpy as np
import pytest

import downsample


def test_input_wrong_threshold():
    """Test the down sampling with wrong input types for x or/and y"""
    x = np.linspace(0.0, 10.0, 100000)
    y = np.linspace(0.0, 10.0, 100000)
    with pytest.raises(TypeError):
        downsample.ltd(x, y, "str")


def test_single_array():
    """Test that the ltd algorithm rejects arrays with multiple dims"""
    x = np.linspace(0, 1, 100000)
    y = np.random.rand(100000)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 100)
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert nx.shape == (100,)
    assert ny.shape == (100,)
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2


def test_negative_threshold():
    """Test if a negative threshold provides problems"""
    x = np.arange(100000, dtype=np.int32)
    y = np.random.randint(1000, size=100000, dtype=np.uint64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    with pytest.raises(ValueError):
        downsample.ltd(x, y, -100)


def test_threshold_larger():
    """Test if a larger threshold provides problems"""
    x = np.arange(100000, dtype=np.int32)
    y = np.random.randint(1000, size=100000, dtype=np.uint64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    # Will return the arrays!
    nx, ny = downsample.ltd(x, y, 100000 + 1)
    assert len(nx) == 100000
    assert len(ny) == 100000
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2


def test_input_list():
    """Test the down sampling with lists types"""
    x = list(range(100000))
    y = [True] * 100000
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([0.0, 1.0, 12500.0, 25000.0, 37500.0,
                           50000.0, 62499.0, 74999.0, 87499.0, 99999.0],
                          dtype=np.double)
    np.testing.assert_array_almost_equal(nx, test_array)
    test_array = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                          dtype=np.double)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_input_list_array():
    """Test the down sampling with mixed types"""
    x = list(range(100000))
    y = np.array([True] * 100000, dtype=bool)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 100)
    assert len(nx) == 100
    assert len(ny) == 100
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([1.0] * 100, dtype=np.double)
    test_array_bool = np.array([1.0] * 100, dtype=bool)
    np.testing.assert_array_almost_equal(ny, test_array)
    np.testing.assert_array_almost_equal(ny, test_array_bool)


def test_array_size():
    """Test the input failure for different dimensions of arrays"""
    x = np.arange(100000)
    y = np.random.randint(1000, size=100000 - 1, dtype=np.uint64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    with pytest.raises(ValueError):
        assert downsample.ltd(x, y, 100000)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2


def test_ltd_uint64():
    """Test the base down sampling of the module"""
    x = np.arange(100000, dtype=np.int32)
    y = np.random.randint(1000, size=100000, dtype=np.uint64)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 100)
    assert len(nx) == 100
    assert len(ny) == 100
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2


def test_ltd_bool():
    """Test the down sampling with boolean types"""
    x = np.arange(100000, dtype=np.int32)
    y = np.array([True] * 100000, dtype=bool)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 100)
    assert len(nx) == 100
    assert len(ny) == 100
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([1.0] * 100, dtype=np.double)
    test_array_bool = np.array([1.0] * 100, dtype=bool)
    np.testing.assert_array_almost_equal(ny, test_array)
    np.testing.assert_array_almost_equal(ny, test_array_bool)


def test_inf():
    """Test the down sampling with inf types"""
    x = np.arange(100000, dtype=np.int32)
    y = np.array([np.inf] * 100000, dtype=np.double)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 100)
    assert len(nx) == 100
    assert len(ny) == 100
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([np.inf] * 100, dtype=np.double)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_single_inf():
    """Test single 'inf' input for down sampling"""
    x = np.arange(20, dtype=np.int32)
    y = np.array(
        [1.0, 1.0, 2.0, np.inf, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
         11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18, 19.0, ],
        dtype=np.double, )
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([0., 0., 4., 5., 7., 10., 12., 14., 16., 19.],
                          dtype=np.double)
    np.testing.assert_array_almost_equal(nx, test_array)
    test_array = np.array([1., 1., 4., 5., 7., 10., 12., 14., 16., 19.],
                          dtype=np.double)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_nan():
    """Test the down sampling with NaN types"""
    x = np.arange(100000, dtype=np.int32)
    y = np.array([np.nan] * 100000, dtype=np.double)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 100)
    assert len(nx) == 100
    assert len(ny) == 100
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2


def test_array_mix_inf_nan():
    """Test mix of problematic input 'inf' and 'nan'"""
    x = np.arange(20, dtype=np.int32)
    y = np.array(
        [0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, np.inf, np.inf,
         10.0, np.nan, 12.0, -np.inf, 14.0, 15.0, 16.0, 17.0, np.nan, 19.0, ],
        dtype=np.double)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    nx, ny = downsample.ltd(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array(
        [0., 0., 4., 4., 6., np.inf, np.nan, -np.inf, 15., 19.],
        dtype=np.double)
    np.testing.assert_array_almost_equal(ny, test_array)


def test_single_nan():
    """Test single 'nan' input for down sampling"""
    x = np.arange(20, dtype=np.double)
    y = np.array([0.0, 1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0,
                  9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                  17.0, 18, 19.0, ], dtype=np.double)
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    z = x.copy()
    nx, ny = downsample.ltd(x, y, 10)
    assert len(nx) == 10
    assert len(ny) == 10
    assert nx.dtype == np.double
    assert ny.dtype == np.double
    assert sys.getrefcount(x) == 2
    assert sys.getrefcount(y) == 2
    assert sys.getrefcount(nx) == 2
    assert sys.getrefcount(ny) == 2
    test_array = np.array([0., 0., 4., 5., 7., 10., 12., 14., 16., 19.],
                          dtype=np.double)
    np.testing.assert_array_almost_equal(x, z)
    np.testing.assert_array_almost_equal(ny, test_array)
