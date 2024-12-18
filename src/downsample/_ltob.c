#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

#include "utils.h"


static PyObject *largest_triangle_one_bucket(PyObject *self, PyObject *args) {
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
    if (!x_array || !y_array)
        goto fail;

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
        // as they are.
        // Special case if the length of points.
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

    // Add the first point and last
    result_x[0] = npy_isfinite(x[0]) ? x[0] : 0.0;
    result_y[0] = npy_isfinite(y[0]) ? y[0] : 0.0;
    result_x[threshold - 1] =
        npy_isfinite(x[len_points - 1]) ? x[len_points - 1] : 0.0;
    result_y[threshold - 1] =
        npy_isfinite(y[len_points - 1]) ? y[len_points - 1] : 0.0;

    double bucket_size = (double)(len_points - 2) / (double)(threshold - 2);

    Py_BEGIN_ALLOW_THREADS;
    // Main loop
    for (npy_intp i = 1; i < threshold - 1; i++) {
        npy_intp start_index = (npy_intp)floor((double)i * bucket_size);
        npy_intp end_index =
            (npy_intp)fmin(len_points - 1, (double)(i + 1) * bucket_size);

        double max_area = -1.0;
        npy_intp max_area_index = -1;
        for (npy_intp j = start_index; j < end_index; j++) {
            double prev_data[2] = {(double)x[j - 1], (double)y[j - 1]};
            double curr_data[2] = {(double)x[j], (double)y[j]};
            double next_data[2] = {(double)x[j + 1], (double)y[j + 1]};
            double area =
                calculate_triangle_area(prev_data, curr_data, next_data);
            if (area > max_area) {
                max_area = area;
                max_area_index = j;
            }
        }
        result_x[i] = (double)x[max_area_index];
        result_y[i] = (double)y[max_area_index];
    }
    Py_END_ALLOW_THREADS;

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

static PyMethodDef LTOBMethods[] = {{"largest_triangle_one_bucket",
                                     largest_triangle_one_bucket, METH_VARARGS,
                                     "Largest triangle one bucket"},
                                    {NULL, NULL, 0, NULL}};

static struct PyModuleDef LTOBModule = {
    PyModuleDef_HEAD_INIT, "_ltob",
    "Module for LTOB downsampling using the NumPy C API", -1, LTOBMethods};

PyMODINIT_FUNC PyInit__ltob(void) {
    import_array();
    return PyModule_Create(&LTOBModule);
}
