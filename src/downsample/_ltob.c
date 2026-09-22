#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>

static inline double calc_triangle_area(double ax, double ay, double bx,
                                        double by, double cx, double cy) {
    return fabs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 0.5);
}

// ==========================================
// Core LTOB
// ==========================================

static void run_ltob(const double *x, const double *y, npy_intp len_points,
                     int threshold, double *result_x, double *result_y) {

    // Always add the first point (handling potential NaNs/Infs)
    result_x[0] = npy_isfinite(x[0]) ? x[0] : 0.0;
    result_y[0] = npy_isfinite(y[0]) ? y[0] : 0.0;

    // Calculate bucket size.
    // We partition the inner points: (len_points - 2) over (threshold - 2)
    // buckets.
    const double bucket_size =
        (double)(len_points - 2) / (double)(threshold - 2);

    // Main loop for the inner buckets
    for (npy_intp i = 1; i < threshold - 1; i++) {

        // Corrected indexing: use (i - 1) so the first iteration (i=1) starts
        // at index 1. This ensures no data points are skipped.
        npy_intp start_index = (npy_intp)(floor((i - 1) * bucket_size) + 1);
        npy_intp end_index = (npy_intp)(floor(i * bucket_size) + 1);

        // Clamp the end index safely
        if (end_index >= len_points) {
            end_index = len_points - 1;
        }

        double max_area = -1.0;
        // Default to start_index to prevent accessing -1 if the loop body is
        // bypassed
        npy_intp max_area_index = start_index;

        // Find the point in this bucket that forms the largest triangle
        // with its immediate neighbors (j-1 and j+1)
        for (npy_intp j = start_index; j < end_index; j++) {

            // Boundary safety: ensure j-1 and j+1 don't read out of bounds
            npy_intp prev_idx = (j - 1 >= 0) ? j - 1 : 0;
            npy_intp next_idx = (j + 1 < len_points) ? j + 1 : len_points - 1;

            double area = calc_triangle_area(x[prev_idx], y[prev_idx], x[j],
                                             y[j], x[next_idx], y[next_idx]);

            if (area > max_area) {
                max_area = area;
                max_area_index = j;
            }
        }

        // Save the best point for this bucket
        result_x[i] = x[max_area_index];
        result_y[i] = y[max_area_index];
    }

    // Always add the last point (handling potential NaNs/Infs)
    npy_intp last_idx = len_points - 1;
    result_x[threshold - 1] = npy_isfinite(x[last_idx]) ? x[last_idx] : 0.0;
    result_y[threshold - 1] = npy_isfinite(y[last_idx]) ? y[last_idx] : 0.0;
}


static PyObject *largest_triangle_one_bucket(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    PyArrayObject *x_array = NULL, *y_array = NULL;
    int threshold;

    // Parse input arguments
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

    // Safely convert inputs
    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    y_array = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    if (!x_array || !y_array)
        goto fail;

    // Validation
    if (PyArray_NDIM(x_array) != 1 || PyArray_NDIM(y_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "x and y must be 1-dimensional.");
        goto fail;
    }
    if (!PyArray_SAMESHAPE(x_array, y_array)) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same shape.");
        goto fail;
    }

    npy_intp len_points = PyArray_DIM(x_array, 0);

    // If threshold is greater than or equal to data size, return arrays
    // unchanged
    if (threshold >= len_points || len_points <= 2) {
        PyObject *result = PyTuple_Pack(2, x_array, y_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return result;
    }

    // Extract raw pointers to the input data
    const double *x = (double *)PyArray_DATA(x_array);
    const double *y = (double *)PyArray_DATA(y_array);

    // Allocate memory for the output arrays using NumPy's safe allocator
    npy_intp dims[1] = {threshold};
    PyObject *npx_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *npy_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!npx_obj || !npy_obj) {
        Py_XDECREF(npx_obj);
        Py_XDECREF(npy_obj);
        goto fail;
    }

    // Extract raw pointers to the output arrays
    double *result_x = (double *)PyArray_DATA((PyArrayObject *)npx_obj);
    double *result_y = (double *)PyArray_DATA((PyArrayObject *)npy_obj);

    // ==========================================
    // Release the GIL and do the heavy lifting
    // ==========================================
    Py_BEGIN_ALLOW_THREADS

        run_ltob(x, y, len_points, threshold, result_x, result_y);

    Py_END_ALLOW_THREADS
        // ==========================================

        // Pack the newly filled arrays into a tuple
        PyObject *result = PyTuple_Pack(2, npx_obj, npy_obj);

    // Clean up our local references
    Py_DECREF(x_array);
    Py_DECREF(y_array);
    Py_DECREF(npx_obj);
    Py_DECREF(npy_obj);

    return result;

fail:
    // Jump target for memory cleanup on failure
    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    return NULL;
}

// ==========================================
// Module Definition and Initialization
// ==========================================

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
