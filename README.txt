# downsample: Collection of downsample algorithms for Python (Python using a C implementation)

This packages includes low level implementations written in C-Python of:

- The `Largest Triangle Dynamic Buckets` (`LTD`) downsampling algorithm
- The `Largest Triangle Three Buckets` (`LTTB`) downsampling algorithm
- The `Largest Triangle One Bucket` (`LTOB`) downsampling algorithm

The algorithm of `LTTB` was initially developed in (https://github.com/dgoeries/lttbc.git)
Parts of this code have been translated and refers to the work of:

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

All functions take an input for ``x`` and ``y`` in addition to the ``threshold``:

    from downsample import ltob, lttb, ltd
    import numpy as np

    array_size = 10000
    threshold = 1000

    x = np.arange(array_size, dtype=np.int32)
    y = np.random.randint(1000, size=array_size, dtype=np.uint64)

    x_l = x.tolist()
    y_l = y.tolist()

    for func in {ltd, ltob, lttb}:
        nx, ny = func(x, y, threshold)
        assert len(nx) == threshold
        assert len(ny) == threshold
        assert nx.dtype == np.double
        assert ny.dtype == np.double

        # List data or a mixture is accepted as well!
        nx, ny = func(x_l, y_l, threshold)
        assert len(nx) == threshold
        assert len(ny) == threshold
        assert nx.dtype == np.double
        assert ny.dtype == np.double

## Performance Overview

For a performance overview, a sample of 7.500 data points was analyzed with a
threshold set at 500 points. The performance test was conducted on a single core,
utilizing a base clock speed of 3.70 GHz and 32 MB of L3 cache.

- LTD: 977.2 us
- LTOB: 61.0 us
- LTTB: 63.1 us
