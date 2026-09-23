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
// Core Algorithm
// ==========================================

static void run_lttb(const double *x, const double *y, npy_intp len_points,
                     int threshold, double *result_x, double *result_y) {

    // Bucket size calculation. (Total points - 2 margins) / (Buckets - 2
    // margins)
    const double every = (double)(len_points - 2) / (threshold - 2);

    // Always add the first point (handling potential NaNs/Infs)
    result_x[0] = npy_isfinite(x[0]) ? x[0] : 0.0;
    result_y[0] = npy_isfinite(y[0]) ? y[0] : 0.0;

    npy_intp a = 0; // Index of the selected point from the previous bucket
    npy_intp next_a = 0;

    for (npy_intp i = 0; i < threshold - 2; ++i) {

        // 1. Calculate the average point (centroid) of the *next* bucket
        double avg_x = 0.0;
        double avg_y = 0.0;

        npy_intp avg_start = (npy_intp)(floor((i + 1) * every) + 1);
        npy_intp avg_end = (npy_intp)(floor((i + 2) * every) + 1);

        if (avg_end >= len_points) {
            avg_end = len_points;
        }

        npy_intp avg_length = avg_end - avg_start;

        for (npy_intp j = avg_start; j < avg_end; j++) {
            avg_x += x[j];
            avg_y += y[j];
        }

        // Multiply by inverse instead of dividing (faster on most CPU
        // architectures)
        double inv_length = 1.0 / (double)avg_length;
        avg_x *= inv_length;
        avg_y *= inv_length;

        // 2. Determine the boundaries of the *current* bucket
        npy_intp range_start = (npy_intp)(floor((i + 0) * every) + 1);
        npy_intp range_end = (npy_intp)(floor((i + 1) * every) + 1);

        double max_area = -1.0;
        double max_area_point_x = 0.0;
        double max_area_point_y = 0.0;

        // Anchor point (from previous iteration)
        double ax = x[a];
        double ay = y[a];

        // 3. Find the point in the current bucket that forms the largest
        // triangle
        for (npy_intp k = range_start; k < range_end; k++) {
            double area = calc_triangle_area(ax, ay, x[k], y[k], avg_x, avg_y);
            if (area > max_area) {
                max_area = area;
                max_area_point_x = x[k];
                max_area_point_y = y[k];
                next_a = k; // Save index to use as anchor for the next bucket
            }
        }

        // Store the selected point
        result_x[i + 1] = max_area_point_x;
        result_y[i + 1] = max_area_point_y;

        // Update anchor index for the next bucket
        a = next_a;
    }

    // Always add the last point (handling potential NaNs/Infs)
    npy_intp last_idx = len_points - 1;
    result_x[threshold - 1] = npy_isfinite(x[last_idx]) ? x[last_idx] : 0.0;
    result_y[threshold - 1] = npy_isfinite(y[last_idx]) ? y[last_idx] : 0.0;
}

static PyObject *largest_triangle_three_buckets(PyObject *self,
                                                PyObject *args) {
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

    // Ensure inputs are arrays or lists
    if ((!PyArray_Check(x_obj) && !PyList_Check(x_obj)) ||
        (!PyArray_Check(y_obj) && !PyList_Check(y_obj))) {
        PyErr_SetString(PyExc_TypeError, "x and y must be list or ndarray.");
        return NULL;
    }

    // Convert inputs to continuous memory C-arrays safely
    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    y_array = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    if (!x_array || !y_array) {
        PyErr_SetString(PyExc_ValueError,
                        "Failed to convert inputs to NumPy arrays.");
        goto fail;
    }

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

    // If threshold exceeds the number of points, just return original data
    if (threshold >= len_points || len_points <= 2) {
        PyObject *result = PyTuple_Pack(2, x_array, y_array);
        Py_DECREF(x_array);
        Py_DECREF(y_array);
        return result;
    }

    // Extract raw pointers from the input arrays
    const double *x = (double *)PyArray_DATA(x_array);
    const double *y = (double *)PyArray_DATA(y_array);

    // Allocate memory for the output array directly through NumPy
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

    Py_BEGIN_ALLOW_THREADS

        run_lttb(x, y, len_points, threshold, result_x, result_y);

    Py_END_ALLOW_THREADS

        Py_DECREF(x_array);

    Py_DECREF(y_array);

    // Pack the results and return them
    PyObject *result = PyTuple_Pack(2, npx_obj, npy_obj);
    Py_DECREF(npx_obj);
    Py_DECREF(npy_obj);
    return result;

fail:
    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    return NULL;
}

// ==========================================
// Module Definition and Initialization
// ==========================================

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
