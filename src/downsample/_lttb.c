#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "utils.h"
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>


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
        // as they are. Special case if the length of points.
        PyObject *result = PyTuple_Pack(2, x_array, y_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return result;
    }
    double *x = (double *)PyArray_DATA(x_array);
    double *y = (double *)PyArray_DATA(y_array);
    // Create an empty output array with shape and dim for the output!
    npy_intp dims[1];
    dims[0] = threshold;
    PyArrayObject *result_x = (PyArrayObject *)PyArray_Empty(
        1, dims, PyArray_DescrFromType(NPY_DOUBLE), 0);
    PyArrayObject *result_y = (PyArrayObject *)PyArray_Empty(
        1, dims, PyArray_DescrFromType(NPY_DOUBLE), 0);

    // Get a pointer to its data
    double *result_x_data = (double *)PyArray_DATA(result_x);
    double *result_y_data = (double *)PyArray_DATA(result_y);

    // The main loop here!
    const double every = (double)(len_points - 2) / (threshold - 2);

    npy_intp a = 0;
    npy_intp next_a = 0;

    double max_area_point_x = 0.0;
    double max_area_point_y = 0.0;

    // Always add the first point!
    result_x_data[0] = npy_isfinite(x[a]) ? x[a] : 0.0;
    result_y_data[0] = npy_isfinite(y[a]) ? y[a] : 0.0;

    for (npy_intp i = 0; i < threshold - 2; ++i) {
        // Calculate point average for next bucket (containing c)
        double avg_x = 0;
        double avg_y = 0;
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
        npy_intp k = (npy_intp)(floor((i + 0) * every) + 1);
        npy_intp range_to = (npy_intp)(floor((i + 1) * every) + 1);

        // Point a
        double point_a_x = x[a];
        double point_a_y = y[a];

        double max_area = -1.0;
        for (; k < range_to; k++) {
            // Calculate triangle area over three buckets
            double point_data[2] = {point_a_x, point_a_y};
            double avg_data[2] = {avg_y, avg_y};
            double next_data[2] = {x[k], y[k]};
            double area =
                calculate_triangle_area(point_data, avg_data, next_data);
            if (area > max_area) {
                max_area = area;
                max_area_point_x = x[k];
                max_area_point_y = y[k];
                next_a = k; // Next a is this b
            }
        }
        // Pick this point from the bucket
        result_x_data[(npy_intp)i + 1] = max_area_point_x;
        result_y_data[(npy_intp)i + 1] = max_area_point_y;
        // Current a becomes the next_a (chosen b)
        a = next_a;
    }

    // Always add last! Check for finite values!
    double last_a_x = x[len_points - 1];
    double last_a_y = y[len_points - 1];
    result_x_data[threshold - 1] = npy_isfinite(last_a_x) ? last_a_x : 0.0;
    result_y_data[threshold - 1] = npy_isfinite(last_a_y) ? last_a_y : 0.0;

    PyObject *value = Py_BuildValue("OO", result_x, result_y);

    // And remove the references!
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_XDECREF(result_x);
    Py_XDECREF(result_y);

    return value;

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
    Py_Initialize();
    import_array();
    return PyModule_Create(&LTTBModule);
}
