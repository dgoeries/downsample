# downsample: Collection of downsample algorithms for Python (Python using a C implementation)

This packages includes:

- A low-level implementation of the `Largest Triangle Dynamic Buckets` (LTD) downsampling algorithm written in C-Python.

The code has been translated and refers to the work of:

- Ján Jakub Naništa (https://github.com/janjakubnanista/downsample) (Typescript)
- Hao Chen (https://github.com/haoel/downsampling) (Go)

The algorithms are described in the work of Sveinn Steinarsson (https://github.com/sveinn-steinarsson/flot-downsample/).

Known features and requirements:

- The algorithm requires that x data is increasing and finite.
- y data must be finite; otherwise, issues may arise.
- x and y data must have the same length.
- The downsample algorithm returns a tuple of two arrays with data type *double*

## Installing

You can also install it [from PyPI](https://pypi.org/project/downsample/)
to use in other environments with Python 3.10 or later:

    pip install downsample

## How to use on the field

The ``ltd`` function takes an input for ``x`` and ``y`` in addition to the ``threshold``:

    import downsample
    import numpy as np

    array_size = 10000
    threshold = 1000

    x = np.arange(array_size, dtype=np.int32)
    y = np.random.randint(1000, size=array_size, dtype=np.uint64)
    nx, ny = downsample.ltd(x, y, threshold)

    assert len(nx) == threshold
    assert len(ny) == threshold
    assert nx.dtype == np.double
    assert ny.dtype == np.double

    # List data or a mixture is accepted as well!
    x = np.arange(array_size).tolist()
    y = np.random.uniform(0, 1000, array_size).tolist()

    assert isinstance(x, list)
    assert isinstance(y, list)

    nx, ny = downsample.ltd(x, y, threshold)

    assert len(nx) == threshold
    assert len(ny) == threshold
    assert nx.dtype == np.double
    assert ny.dtype == np.double
