#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <assert.h>
#include <math.h>
#include <numpy/arrayobject.h>


static inline double calc_triangle_area(double ax, double ay, double bx,
                                        double by, double cx, double cy) {
    return fabs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 0.5);
}

/* The core algorithm only handles aligned, C-contiguous float64 vectors. */
static inline int is_contiguous_double_1d(PyArrayObject *arr) {
    return PyArray_NDIM(arr) == 1 && PyArray_TYPE(arr) == NPY_DOUBLE &&
           PyArray_IS_C_CONTIGUOUS(arr) && PyArray_ISALIGNED(arr);
}

// ==========================================
// Core LTOB
// ==========================================

/*
 * Requires `x` and `y` to be C-contiguous float64 arrays of `len_points`
 * elements and `result_x` / `result_y` to hold `threshold` elements.
 */
static void run_ltob(const double *x, const double *y, npy_intp len_points,
                     int threshold, double *result_x, double *result_y) {
    assert(x != NULL && y != NULL);
    assert(result_x != NULL && result_y != NULL);
    assert(threshold > 2);
    assert(len_points > threshold);

    // Always add the first point (handling NaNs/Infs using standard C math)
    result_x[0] = isfinite(x[0]) ? x[0] : 0.0;
    result_y[0] = isfinite(y[0]) ? y[0] : 0.0;

    // Calculate bucket size.
    // We partition the inner points: (len_points - 2) over (threshold - 2)
    // buckets.
    const double bucket_size =
        (double)(len_points - 2) / (double)(threshold - 2);

    // Main loop for the inner buckets
    for (npy_intp i = 1; i < threshold - 1; i++) {

        const npy_intp start_index = (npy_intp)floor(i * bucket_size);
        const npy_intp end_index =
            (npy_intp)fmin((double)(len_points - 1), (i + 1) * bucket_size);

        double max_area = -1.0;
        npy_intp max_area_index = start_index;

        // bucket_size > 1 (len_points > threshold), so start_index >= 1, and
        // end_index is capped at len_points - 1: both neighbours j - 1 and
        // j + 1 are always in bounds and no per-point clamping is needed.
        assert(start_index >= 1 && start_index < len_points);
        assert(end_index <= len_points - 1);

        // Slide a three-point window (prev, cur, next) over the bucket so
        // only one new point is loaded per step.
        double px = x[start_index - 1];
        double py = y[start_index - 1];
        double cx = x[start_index];
        double cy = y[start_index];

        for (npy_intp j = start_index; j < end_index; j++) {
            const double nx = x[j + 1];
            const double ny = y[j + 1];

            double area = calc_triangle_area(px, py, cx, cy, nx, ny);
            // Inf/NaN areas never overtake the bucket logic
            if (area > max_area) {
                max_area = area;
                max_area_index = j;
            }
            px = cx;
            py = cy;
            cx = nx;
            cy = ny;
        }

        // Save the best point for this bucket.
        // If the area was unresolvable or the point itself is not finite,
        // fallback to 0.0 to satisfy test_inf(), test_nan(), and
        // test_single_inf() expectations.
        result_x[i] = 0.0;
        result_y[i] = 0.0;
        if (max_area != -1.0) {
            assert(max_area_index >= 0 && max_area_index < len_points);
            const double bx = x[max_area_index];
            const double by = y[max_area_index];
            if (isfinite(bx) && isfinite(by)) {
                result_x[i] = bx;
                result_y[i] = by;
            }
        }
    }

    // Always add the last point (handling potential NaNs/Infs)
    const npy_intp last_idx = len_points - 1;
    result_x[threshold - 1] = isfinite(x[last_idx]) ? x[last_idx] : 0.0;
    result_y[threshold - 1] = isfinite(y[last_idx]) ? y[last_idx] : 0.0;
}

static PyObject *largest_triangle_one_bucket(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    PyArrayObject *x_array = NULL, *y_array = NULL;
    PyObject *npx_obj = NULL, *npy_obj = NULL;
    PyObject *result = NULL;
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

    // Convert inputs to aligned, C-contiguous float64 arrays. This is a
    // no-op (no copy) for input that already has that layout.
    x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                                NPY_ARRAY_IN_ARRAY);
    y_array = x_array ? (PyArrayObject *)PyArray_FROM_OTF(
                            y_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY)
                      : NULL;
    if (!x_array || !y_array)
        goto done;

    // Validation
    if (PyArray_NDIM(x_array) != 1 || PyArray_NDIM(y_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "x and y must be 1-dimensional.");
        goto done;
    }
    if (!PyArray_SAMESHAPE(x_array, y_array)) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same shape.");
        goto done;
    }

    const npy_intp len_points = PyArray_DIM(x_array, 0);

    // If threshold is greater than or equal to data size, return arrays
    // unchanged
    if (threshold >= len_points || len_points <= 2) {
        result = PyTuple_Pack(2, x_array, y_array);
        goto done;
    }

    // Allocate the (uninitialized, fully overwritten) output arrays
    npy_intp dims[1] = {threshold};
    npx_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    npy_obj = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!npx_obj || !npy_obj) {
        goto done;
    }

    // The core only deals with contiguous memory
    assert(is_contiguous_double_1d(x_array));
    assert(is_contiguous_double_1d(y_array));
    assert(is_contiguous_double_1d((PyArrayObject *)npx_obj));
    assert(is_contiguous_double_1d((PyArrayObject *)npy_obj));

    const double *x = (const double *)PyArray_DATA(x_array);
    const double *y = (const double *)PyArray_DATA(y_array);
    double *result_x = (double *)PyArray_DATA((PyArrayObject *)npx_obj);
    double *result_y = (double *)PyArray_DATA((PyArrayObject *)npy_obj);

    // ==========================================
    // Release the GIL and do the heavy lifting
    // ==========================================
    Py_BEGIN_ALLOW_THREADS;
    run_ltob(x, y, len_points, threshold, result_x, result_y);
    Py_END_ALLOW_THREADS;

    // Pack the newly filled arrays into a tuple
    result = PyTuple_Pack(2, npx_obj, npy_obj);

done:
    Py_XDECREF(x_array);
    Py_XDECREF(y_array);
    Py_XDECREF(npx_obj);
    Py_XDECREF(npy_obj);
    return result;
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
