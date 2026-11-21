#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <assert.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>


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
// Core Algorithm
// ==========================================

/*
 * Requires `x` and `y` to be C-contiguous float64 arrays of `len_points`
 * elements and `result_x` / `result_y` to hold `threshold` elements.
 */
static void run_lttb(const double *x, const double *y, npy_intp len_points,
                     int threshold, double *result_x, double *result_y) {
    assert(x != NULL && y != NULL);
    assert(result_x != NULL && result_y != NULL);
    assert(threshold > 2);
    assert(len_points > threshold);

    // Bucket size calculation. (Total points - 2 margins) / (Buckets - 2
    // margins)
    const double every = (double)(len_points - 2) / (threshold - 2);

    // Always add the first point (handling potential NaNs/Infs)
    result_x[0] = npy_isfinite(x[0]) ? x[0] : 0.0;
    result_y[0] = npy_isfinite(y[0]) ? y[0] : 0.0;

    npy_intp a = 0; // Index of the selected point from the previous bucket

    // Boundaries of the *current* bucket. The end of the current bucket is
    // identical to the start of the next (averaging) bucket, so each
    // boundary is computed only once and carried over to the next iteration.
    npy_intp range_start = (npy_intp)(floor((npy_intp)0 * every) + 1);
    npy_intp range_end = (npy_intp)(floor((npy_intp)1 * every) + 1);

    for (npy_intp i = 0; i < threshold - 2; ++i) {

        // 1. Calculate the average point (centroid) of the *next* bucket
        const npy_intp avg_start = range_end;
        const npy_intp next_range_end =
            (npy_intp)(floor((i + 2) * every) + 1);
        npy_intp avg_end = next_range_end;

        if (avg_end >= len_points) {
            avg_end = len_points;
        }

        double avg_x = 0.0;
        double avg_y = 0.0;
        for (npy_intp j = avg_start; j < avg_end; j++) {
            avg_x += x[j];
            avg_y += y[j];
        }

        // Multiply by inverse instead of dividing
        const double inv_length = 1.0 / (double)(avg_end - avg_start);
        avg_x *= inv_length;
        avg_y *= inv_length;

        // 2. Find the point in the current bucket that forms the largest
        // triangle with the previous anchor and the next bucket's centroid.
        assert(range_start >= 1 && range_end <= len_points);
        const double ax = x[a];
        const double ay = y[a];

        double max_area = -1.0;
        npy_intp best = -1;

        for (npy_intp k = range_start; k < range_end; k++) {
            double area = calc_triangle_area(ax, ay, x[k], y[k], avg_x, avg_y);
            if (area > max_area) {
                max_area = area;
                best = k;
            }
        }

        // 3. Store the selected point. If no area was comparable (NaN), the
        // point is reported as 0.0 and the anchor stays unchanged.
        if (best >= 0) {
            result_x[i + 1] = x[best];
            result_y[i + 1] = y[best];
            a = best;
        } else {
            result_x[i + 1] = 0.0;
            result_y[i + 1] = 0.0;
        }

        // Advance to the next bucket
        range_start = avg_start;
        range_end = next_range_end;
    }

    // Always add the last point (handling potential NaNs/Infs)
    const npy_intp last_idx = len_points - 1;
    result_x[threshold - 1] = npy_isfinite(x[last_idx]) ? x[last_idx] : 0.0;
    result_y[threshold - 1] = npy_isfinite(y[last_idx]) ? y[last_idx] : 0.0;
}

static PyObject *largest_triangle_three_buckets(PyObject *self,
                                                PyObject *args) {
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

    // Ensure inputs are arrays or lists
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
    if (!x_array || !y_array) {
        PyErr_SetString(PyExc_ValueError,
                        "Failed to convert inputs to NumPy arrays.");
        goto done;
    }

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

    // If threshold exceeds the number of points, just return original data
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

    Py_BEGIN_ALLOW_THREADS;
    run_lttb(x, y, len_points, threshold, result_x, result_y);
    Py_END_ALLOW_THREADS;

    // Pack the results and return them
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
