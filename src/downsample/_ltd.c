#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <assert.h>
#include <float.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    npy_intp start; // inclusive index in the points array
    npy_intp end;   // exclusive index in the points array

    // Memoized SSE of this bucket together with the exact inputs it was
    // computed from. The SSE only depends on the bucket range and on its two
    // anchor points, so it is reused as long as none of those changed. The
    // cache travels with the bucket when buckets are shifted.
    npy_intp key_prev;  // index of the previous anchor point
    npy_intp key_start; // bucket start at computation time (-1: invalid)
    npy_intp key_end;   // bucket end at computation time
    npy_intp key_next;  // index of the next anchor point
    double sse;
} BucketRange;


static inline double calc_triangle_area(double ax, double ay, double bx,
                                        double by, double cx, double cy) {
    return fabs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 0.5);
}

/* The core algorithm only handles aligned, C-contiguous float64 vectors. */
static inline int is_contiguous_double_1d(PyArrayObject *arr) {
    return PyArray_NDIM(arr) == 1 && PyArray_TYPE(arr) == NPY_DOUBLE &&
           PyArray_IS_C_CONTIGUOUS(arr) && PyArray_ISALIGNED(arr);
}

/*
 * Calculates the Sum of Squared Errors (SSE) for a given bucket by fitting a
 * linear regression line. It includes the previously selected point (prev)
 * and the next anchor point (next) to ensure continuity.
 */
static inline double calculate_sse_bucket(const double *x, const double *y,
                                          npy_intp start, npy_intp end,
                                          double prev_x, double prev_y,
                                          double next_x, double next_y) {
    assert(start >= 0 && start <= end);

    const npy_intp bucket_len = end - start;
    const npy_intp total_points = bucket_len + 2; // Bucket + prev + next

    double sum_x = prev_x + next_x;
    double sum_y = prev_y + next_y;

    // Accumulate sums for X and Y to find the mean (centroid)
    for (npy_intp i = start; i < end; i++) {
        sum_x += x[i];
        sum_y += y[i];
    }

    const double avg_x = sum_x / (double)total_points;
    const double avg_y = sum_y / (double)total_points;

    double numerator = 0.0;
    double denominator = 0.0;

    // Calculate variance/covariance for the previous point
    double dx = prev_x - avg_x;
    double dy = prev_y - avg_y;
    numerator += dx * dy;
    denominator += dx * dx;

    // Calculate variance/covariance for all points in the bucket
    for (npy_intp i = start; i < end; i++) {
        dx = x[i] - avg_x;
        dy = y[i] - avg_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    // Calculate variance/covariance for the next point
    dx = next_x - avg_x;
    dy = next_y - avg_y;
    numerator += dx * dy;
    denominator += dx * dx;

    // Calculate linear regression coefficients: y = ax + b
    double a, b;
    if (denominator == 0.0) {
        // Handle vertical line edge-case
        a = (numerator > 0) ? INFINITY : -INFINITY;
        b = avg_y;
    } else {
        a = numerator / denominator;
        b = avg_y - a * avg_x; // y-intercept
    }

    // Sum of Squared Errors (SSE) of the vertical residuals against the
    // regression line, including both anchor points.
    double sse = 0.0;

    double err = prev_y - (a * prev_x + b);
    sse += err * err;

    for (npy_intp i = start; i < end; i++) {
        // err = (Actual Y) - (Predicted Y on the trendline)
        err = y[i] - (a * x[i] + b);
        sse += err * err;
    }

    err = next_y - (a * next_x + b);
    sse += err * err;

    return sse;
}

// ==========================================
// Core Algorithm
// ==========================================

/*
 * Requires `x` and `y` to be C-contiguous float64 arrays of `num_points`
 * elements, `out_x` / `out_y` to hold `threshold` elements and `buckets` to
 * provide room for `threshold + 1` entries (one scratch slot used while
 * splitting). The function cannot fail.
 */
static void run_ltd(const double *x, const double *y, npy_intp num_points,
                    int threshold, BucketRange *buckets, double *out_x,
                    double *out_y) {
    assert(x != NULL && y != NULL);
    assert(out_x != NULL && out_y != NULL && buckets != NULL);
    assert(threshold > 2);
    assert(num_points > threshold);

    for (int i = 0; i <= threshold; i++) {
        buckets[i].key_start = -1; // no cached SSE yet
    }

    // The first and last buckets are explicitly fixed to the first and last
    // points
    buckets[0].start = 0;
    buckets[0].end = 1;

    // Distribute the remaining points evenly across the middle buckets
    const double bucket_size = (double)(num_points - 2) / (threshold - 2);
    for (int i = 0; i < threshold - 2; i++) {
        buckets[i + 1].start = (npy_intp)floor(i * bucket_size) + 1;
        buckets[i + 1].end = (npy_intp)floor((i + 1) * bucket_size) + 1;
    }

    buckets[threshold - 1].start = num_points - 1;
    buckets[threshold - 1].end = num_points;

    // Dynamic optimization iterations to adjust bucket boundaries
    const int num_iterations = 10;

    for (int iter = 0; iter < num_iterations; iter++) {
        // Step 1. Calculate SSE for all middle buckets. A split / merge only
        // touches a handful of buckets, so after the first iteration nearly
        // all values are served from the per-bucket cache instead of
        // re-scanning every point (identical results, O(n / threshold) work
        // per iteration instead of O(n)).
        for (int i = 1; i < threshold - 1; i++) {
            BucketRange *bk = &buckets[i];
            // Anchor points are the last point of previous bucket and first
            // of next bucket
            const npy_intp prev_idx = buckets[i - 1].end - 1;
            const npy_intp next_idx = buckets[i + 1].start;
            assert(prev_idx >= 0 && prev_idx < num_points);
            assert(next_idx >= 0 && next_idx < num_points);
            assert(bk->end <= num_points);

            if (bk->key_start != bk->start || bk->key_end != bk->end ||
                bk->key_prev != prev_idx || bk->key_next != next_idx) {
                bk->sse = calculate_sse_bucket(x, y, bk->start, bk->end,
                                               x[prev_idx], y[prev_idx],
                                               x[next_idx], y[next_idx]);
                bk->key_start = bk->start;
                bk->key_end = bk->end;
                bk->key_prev = prev_idx;
                bk->key_next = next_idx;
            }
        }

        // Step 2. Find the bucket with the highest SSE (must have >1 point to
        // split)
        double max_sse = 0.0;
        int high_idx = -1;
        for (int i = 1; i < threshold - 1; i++) {
            const npy_intp b_len = buckets[i].end - buckets[i].start;
            if (b_len > 1 && buckets[i].sse > max_sse) {
                max_sse = buckets[i].sse;
                high_idx = i;
            }
        }

        // If we can't find a bucket to split, optimization is complete
        if (high_idx < 0)
            break;

        // Step 3. Find the pair of adjacent buckets with the lowest combined
        // SSE (We exclude the bucket we just marked for splitting)
        double min_sum = INFINITY;
        int low_idx = -1;
        for (int i = 1; i < threshold - 2; i++) {
            if (i == high_idx || (i + 1) == high_idx)
                continue;

            const double sum = buckets[i].sse + buckets[i + 1].sse;
            if (sum < min_sum) {
                min_sum = sum;
                low_idx = i;
            }
        }

        // If we can't find buckets to merge, optimization is complete
        if (low_idx < 0)
            break;

        assert(high_idx >= 1 && high_idx <= threshold - 2);
        assert(low_idx >= 1 && low_idx <= threshold - 3);

        // Step 4. Split the highest SSE bucket perfectly in half.
        const npy_intp total_len =
            buckets[high_idx].end - buckets[high_idx].start;
        const npy_intp half = (npy_intp)ceil((double)total_len / 2.0);
        const npy_intp orig_end = buckets[high_idx].end;

        // Shift buckets [high_idx + 1, threshold) one slot to the right to
        // make room for the newly split bucket (uses the scratch slot).
        memmove(&buckets[high_idx + 2], &buckets[high_idx + 1],
                sizeof(BucketRange) * (size_t)(threshold - high_idx - 1));

        // Assign boundaries for the two newly split halves. The cache keys
        // no longer match, so both halves are recomputed.
        buckets[high_idx].end = buckets[high_idx].start + half;
        buckets[high_idx + 1].start = buckets[high_idx].end;
        buckets[high_idx + 1].end = orig_end;
        buckets[high_idx + 1].key_start = -1;

        // Step 5. Merge the pair selected before the split. A split to its
        // left shifted the pair one position to the right.
        if (low_idx > high_idx) {
            low_idx++;
        }

        buckets[low_idx].end = buckets[low_idx + 1].end;
        memmove(&buckets[low_idx + 1], &buckets[low_idx + 2],
                sizeof(BucketRange) * (size_t)(threshold - low_idx - 1));
    }

    // ==========================================
    // LTTB Point Selection Phase
    // ==========================================
    // Now that buckets are optimally sized, select the best point from each
    // bucket.

    // Always select the exact first point
    out_x[0] = x[0];
    out_y[0] = y[0];
    double last_x = out_x[0];
    double last_y = out_y[0];

    // Iterate over the middle buckets to pick the point that forms the largest
    // triangle
    for (int i = 1; i < threshold - 1; i++) {

        // 1. Calculate the average point (centroid) of the *next* bucket
        const npy_intp next_start = buckets[i + 1].start;
        const npy_intp next_end = buckets[i + 1].end;
        const npy_intp next_count = next_end - next_start;
        assert(next_start >= 0 && next_end <= num_points);

        double avg_next_x = 0.0, avg_next_y = 0.0;
        for (npy_intp k = next_start; k < next_end; k++) {
            avg_next_x += x[k];
            avg_next_y += y[k];
        }
        avg_next_x /= (double)next_count;
        avg_next_y /= (double)next_count;

        // 2. Find the point in the *current* bucket that forms the largest
        // triangle using the last selected point and the next bucket's average
        // point
        double max_area = -1.0;
        // The original array-view implementation addressed index -1 when no
        // area was comparable (for example because it was NaN), which selects
        // the point immediately before this contiguous bucket.
        npy_intp best_idx = buckets[i].start - 1;

        for (npy_intp j = buckets[i].start; j < buckets[i].end; j++) {
            const double area = calc_triangle_area(last_x, last_y, x[j], y[j],
                                                   avg_next_x, avg_next_y);
            if (area > max_area) {
                max_area = area;
                best_idx = j;
            }
        }

        // Store the selected best point and update 'last_x / last_y' for the
        // next iteration
        assert(best_idx >= 0 && best_idx < num_points);
        last_x = x[best_idx];
        last_y = y[best_idx];
        out_x[i] = last_x;
        out_y[i] = last_y;
    }

    // Always select the exact last point
    out_x[threshold - 1] = x[num_points - 1];
    out_y[threshold - 1] = y[num_points - 1];
}


static PyObject *largest_triangle_dynamic(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    PyArrayObject *x = NULL, *y = NULL;
    PyObject *out_x_arr = NULL, *out_y_arr = NULL;
    BucketRange *buckets = NULL;
    PyObject *result = NULL;
    int threshold;

    // Parse arguments: (Object, Object, Integer)
    if (!PyArg_ParseTuple(args, "OOi", &x_obj, &y_obj, &threshold)) {
        return NULL; // Return NULL propagates the Python exception
    }

    // Validation
    if (threshold <= 2) {
        PyErr_SetString(PyExc_ValueError, "Threshold must be larger than 2.");
        return NULL;
    }

    // Convert input objects to aligned, C-contiguous float64 arrays. This is
    // a no-op (no copy) for input that already has that layout.
    x = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                          NPY_ARRAY_IN_ARRAY);
    y = x ? (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
                                              NPY_ARRAY_IN_ARRAY)
          : NULL;

    // Check if conversion failed
    if (!x || !y)
        goto done;

    // Ensure they are 1-Dimensional and equal length
    if (PyArray_NDIM(x) != 1 || PyArray_NDIM(y) != 1 ||
        !PyArray_SAMESHAPE(x, y)) {
        PyErr_SetString(PyExc_ValueError,
                        "x and y must be 1D with identical shape.");
        goto done;
    }

    const npy_intp len_points = PyArray_DIM(x, 0);

    // If the data is already smaller than the threshold, just return the data
    // untouched
    if (threshold >= len_points) {
        result = PyTuple_Pack(2, x, y);
        goto done;
    }

    // Allocate the (uninitialized, fully overwritten) output arrays
    npy_intp out_dim = threshold;
    out_x_arr = PyArray_SimpleNew(1, &out_dim, NPY_DOUBLE);
    out_y_arr = PyArray_SimpleNew(1, &out_dim, NPY_DOUBLE);
    if (!out_x_arr || !out_y_arr)
        goto done;

    // Bucket work space (+1 scratch slot for splitting). Allocated while
    // holding the GIL so a failure raises a proper MemoryError instead of
    // silently returning garbage.
    buckets = (BucketRange *)PyMem_Malloc(sizeof(BucketRange) *
                                          ((size_t)threshold + 1));
    if (!buckets) {
        PyErr_NoMemory();
        goto done;
    }

    // The core only deals with contiguous memory
    assert(is_contiguous_double_1d(x));
    assert(is_contiguous_double_1d(y));
    assert(is_contiguous_double_1d((PyArrayObject *)out_x_arr));
    assert(is_contiguous_double_1d((PyArrayObject *)out_y_arr));

    const double *x_raw = (const double *)PyArray_DATA(x);
    const double *y_raw = (const double *)PyArray_DATA(y);
    double *out_x = (double *)PyArray_DATA((PyArrayObject *)out_x_arr);
    double *out_y = (double *)PyArray_DATA((PyArrayObject *)out_y_arr);

    Py_BEGIN_ALLOW_THREADS;
    run_ltd(x_raw, y_raw, len_points, threshold, buckets, out_x, out_y);
    Py_END_ALLOW_THREADS;

    // Pack the output X and Y arrays into a Python tuple and return it
    result = PyTuple_Pack(2, out_x_arr, out_y_arr);

done:
    PyMem_Free(buckets);
    Py_XDECREF(x);
    Py_XDECREF(y);
    Py_XDECREF(out_x_arr);
    Py_XDECREF(out_y_arr);
    return result;
}

// ==========================================
// Module Definition and Initialization
// ==========================================

// Define the methods exposed by this module
static PyMethodDef LTDMethods[] = {
    {"largest_triangle_dynamic", largest_triangle_dynamic,
     METH_VARARGS, // Expect standard arguments (positional)
     "Largest triangle dynamic for buckets"},
    {NULL, NULL, 0, NULL} // Sentinel denoting the end of the method list
};

// Define the module structure
static struct PyModuleDef LTDModule = {
    PyModuleDef_HEAD_INIT,
    "_ltd", // Module name
    "Module for LTD downsampling using the NumPy C API",
    -1, // -1 means global state module (no sub-interpreter state required)
    LTDMethods};

PyMODINIT_FUNC PyInit__ltd(void) {
    import_array();
    return PyModule_Create(&LTDModule);
}
