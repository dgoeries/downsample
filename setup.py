import numpy
from setuptools import Extension, find_packages, setup

_HEADERS = "src/downsample"

setup(
    ext_modules=[
        Extension(
            "downsample._ltd",
            ["src/downsample/_ltd.c", "src/downsample/utils.c"],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[numpy.get_include(), _HEADERS]),
        Extension(
            "downsample._ltob",
            ["src/downsample/_ltob.c", "src/downsample/utils.c"],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[numpy.get_include(), _HEADERS]),
        Extension(
            "downsample._lttb",
            ["src/downsample/_lttb.c", "src/downsample/utils.c"],
            define_macros=[
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            include_dirs=[numpy.get_include(), _HEADERS])
    ],
    package_dir={"": "src"},
    package_data={
        "downsample": ["src/downsample/*.pyi",
                       "src/downsample/py.typed"],
    },
    packages=find_packages("src"),
)
