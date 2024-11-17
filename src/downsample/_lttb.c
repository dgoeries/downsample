#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "utils.h"


static PyObject *largest_triangle_three_buckets(PyObject *self,
                                                PyObject *args) {
    PyObject *x_obj, *y_obj;
    PyArrayObject *x_array = NULL, *y_array = NULL;
    int threshold;

    if (!PyArg_ParseTuple(args, "OOi", &x_obj, &y_obj, &threshold)) {
        return NULL;
    }

    if (threshold <= 2) {
        PyErr_SetString(PyExc_ValueError, "Threshold must be larger than 2.");
        return NULL;
    }

    if ((!PyArray_Check(x_obj) && !PyList_Check(x_obj)) ||
        (!PyArray_Check(y_obj) && !PyList_Check(y_obj))) {
        PyErr_SetString(PyExc_TypeError, "x and y must be list or ndarray.");
        return NULL;
    }

    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    y_array = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    if (!x_array || !y_array) {
        PyErr_SetString(PyExc_ValueError,
                        "Failed to convert inputs to NumPy arrays.");
        goto fail;
    }

    if (PyArray_NDIM(x_array) != 1 || PyArray_NDIM(y_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "x and y must be 1-dimensional.");
        goto fail;
    }
    if (!PyArray_SAMESHAPE(x_array, y_array)) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same shape.");
        goto fail;
    }

    npy_intp len_points = PyArray_DIM(x_array, 0);
    if (threshold >= len_points || len_points <= 2) {
        // If the threshold is greater than the number of points, return x and y
        // as they are. Special case if the length of points.
        PyObject *result = PyTuple_Pack(2, x_array, y_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return result;
    }

    double *x = (double *)PyArray_DATA(x_array);
    double *y = (double *)PyArray_DATA(y_array);

    double *result_x = (double *)malloc(threshold * sizeof(double));
    double *result_y = (double *)malloc(threshold * sizeof(double));
    if (!result_x || !result_y) {
        PyErr_SetString(PyExc_MemoryError,
                        "Failed to allocate memory for result arrays.");
        free(result_x);
        free(result_y);
        goto fail;
    }

    const double every = (double)(len_points - 2) / (threshold - 2);
    // Always add the first point!
    result_x[0] = npy_isfinite(x[0]) ? x[0] : 0.0;
    result_y[0] = npy_isfinite(y[0]) ? y[0] : 0.0;

    npy_intp a = 0, next_a = 0;

    Py_BEGIN_ALLOW_THREADS;
    for (npy_intp i = 0; i < threshold - 2; ++i) {
        double avg_x = 0, avg_y = 0;
        // Careful, thread local variables
        double max_area_point_x = 0.0;
        double max_area_point_y = 0.0;
        npy_intp avg_start = (npy_intp)(floor((i + 1) * every) + 1);
        npy_intp avg_end = (npy_intp)(floor((i + 2) * every) + 1);
        if (avg_end >= len_points) {
            avg_end = len_points;
        }
        npy_intp avg_length = avg_end - avg_start;

        for (; avg_start < avg_end; avg_start++) {
            avg_x += x[avg_start];
            avg_y += y[avg_start];
        }
        avg_x /= avg_length;
        avg_y /= avg_length;

        // Get the range for this bucket
        npy_intp range_start = (npy_intp)(floor((i + 0) * every) + 1);
        npy_intp range_end = (npy_intp)(floor((i + 1) * every) + 1);

        // Point a
        double point_a[2] = {x[a], y[a]};
        double max_area = -1.0;

        for (npy_intp k = range_start; k < range_end; k++) {
            double point_k[2] = {x[k], y[k]};
            double avg_point[2] = {avg_x, avg_y};
            double area = calculate_triangle_area(point_a, avg_point, point_k);
            if (area > max_area) {
                max_area = area;
                max_area_point_x = x[k];
                max_area_point_y = y[k];
                next_a = k;
            }
        }

        result_x[i + 1] = max_area_point_x;
        result_y[i + 1] = max_area_point_y;
        a = next_a;
    }
    Py_END_ALLOW_THREADS;

    result_x[threshold - 1] =
        npy_isfinite(x[len_points - 1]) ? x[len_points - 1] : 0.0;
    result_y[threshold - 1] =
        npy_isfinite(y[len_points - 1]) ? y[len_points - 1] : 0.0;

    npy_intp dims[1] = {threshold};
    PyObject *npx =
        PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void *)result_x);
    PyObject *npy =
        PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, (void *)result_y);
    PyArray_ENABLEFLAGS((PyArrayObject *)npx, NPY_ARRAY_OWNDATA);
    PyArray_ENABLEFLAGS((PyArrayObject *)npy, NPY_ARRAY_OWNDATA);

    PyObject *result = PyTuple_Pack(2, npx, npy);
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(npx);
    Py_DECREF(npy);
    return result;

fail:
    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    return NULL;
}

static PyMethodDef LTTBMethods[] = {
    {"largest_triangle_three_buckets", largest_triangle_three_buckets,
     METH_VARARGS,
     "Compute the largest triangle three buckets (LTTB) algorithm in a C "
     "extension."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef LTTBModule = {
    PyModuleDef_HEAD_INIT, "_lttb",
    "A Python module that computes the largest triangle three buckets "
    "algorithm (LTTB) using C code.",
    -1, LTTBMethods};

PyMODINIT_FUNC PyInit__lttb(void) {
    import_array();
    return PyModule_Create(&LTTBModule);
}
