#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>

#define PY_SSIZE_T_CLEAN
#include <float.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>


typedef struct {
    npy_intp start; // inclusive index in the points array
    npy_intp end;   // exclusive index in the points array
} BucketRange;


static inline double calc_triangle_area(double ax, double ay, double bx,
                                        double by, double cx, double cy) {
    return fabs((ax * (by - cy) + bx * (cy - ay) + cx * (ay - by)) * 0.5);
}

/*
 * Calculates the Sum of Squared Errors (SSE) for a given bucket by fitting a
 * linear regression line. It includes the previously selected point (prev_pt)
 * and the next anchor point (next_pt) to ensure continuity.
 */
static double calculate_sse_bucket(const double *points, npy_intp start,
                                   npy_intp end, const double *prev_pt,
                                   const double *next_pt) {
    npy_intp bucket_len = end - start;
    npy_intp total_points = bucket_len + 2; // Bucket points + prev + next

    double sum_x = prev_pt[0] + next_pt[0];
    double sum_y = prev_pt[1] + next_pt[1];

    // Accumulate sums for X and Y to find the mean (centroid)
    for (npy_intp i = start; i < end; i++) {
        sum_x += points[i * 2];
        sum_y += points[i * 2 + 1];
    }

    double avg_x = sum_x / (double)total_points;
    double avg_y = sum_y / (double)total_points;

    double numerator = 0.0;
    double denominator = 0.0;

    // Calculate variance/covariance for the previous point
    double dx = prev_pt[0] - avg_x;
    double dy = prev_pt[1] - avg_y;
    numerator += dx * dy;
    denominator += dx * dx;

    // Calculate variance/covariance for all points in the bucket
    for (npy_intp i = start; i < end; i++) {
        dx = points[i * 2] - avg_x;
        dy = points[i * 2 + 1] - avg_y;
        numerator += dx * dy;
        denominator += dx * dx;
    }

    // Calculate variance/covariance for the next point
    dx = next_pt[0] - avg_x;
    dy = next_pt[1] - avg_y;
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

    // Calculate the actual Sum of Squared Errors (SSE) against the regression
    // line
    double sse = 0.0;

    // Error of previous point
    double err = prev_pt[1] - (a * prev_pt[0] + b);
    sse += err * err;

    // Iterate through every point inside the current bucket's index range.
    for (npy_intp i = start; i < end; i++) {
        // Calculate the vertical residual (error) between the actual data point
        // and our fitted linear regression line equation: y = ax + b.
        //
        // Memory mapping for the interleaved 'points' array:
        // points[i * 2]     -> The actual X coordinate
        // points[i * 2 + 1] -> The actual Y coordinate
        //
        // Math breakdown:
        // (a * points[i * 2] + b) -> The predicted Y value on the trendline
        // err = (Actual Y) - (Predicted Y)
        err = points[i * 2 + 1] - (a * points[i * 2] + b);
        sse += err * err;
    }

    // Error of next point
    err = next_pt[1] - (a * next_pt[0] + b);
    sse += err * err;

    return sse;
}

// ==========================================
// Core Algorithm
// ==========================================

static void run_ltd(const double *points, npy_intp num_points, int threshold,
                    double *out_x, double *out_y) {

    // Allocate contiguous arrays for buckets and their computed SSE values
    BucketRange *buckets =
        (BucketRange *)malloc(sizeof(BucketRange) * (threshold + 1));
    double *sse = (double *)malloc(sizeof(double) * threshold);

    // Check for memory allocation failure
    if (!buckets || !sse) {
        free(buckets);
        free(sse);
        return;
    }

    // The first and last buckets are explicitly fixed to the first and last
    // points
    buckets[0].start = 0;
    buckets[0].end = 1;

    // Distribute the remaining points evenly across the middle buckets
    double bucket_size = (double)(num_points - 2) / (threshold - 2);
    for (int i = 0; i < threshold - 2; i++) {
        buckets[i + 1].start = (npy_intp)floor(i * bucket_size) + 1;
        buckets[i + 1].end = (npy_intp)floor((i + 1) * bucket_size) + 1;
    }

    buckets[threshold - 1].start = num_points - 1;
    buckets[threshold - 1].end = num_points;

    // Dynamic optimization iterations to adjust bucket boundaries
    int num_iterations =
        10; // Empirically derived: (threshold * 10) / threshold

    for (int iter = 0; iter < num_iterations; iter++) {
        // Step 1. Calculate SSE for all middle buckets
        for (int i = 1; i < threshold - 1; i++) {
            // Anchor points are the last point of previous bucket and first of
            // next bucket
            const double *prev_pt = &points[(buckets[i - 1].end - 1) * 2];
            const double *next_pt = &points[buckets[i + 1].start * 2];

            sse[i] = calculate_sse_bucket(points, buckets[i].start,
                                          buckets[i].end, prev_pt, next_pt);
        }

        // Step 2. Find the bucket with the highest SSE (must have >1 point to
        // split)
        double max_sse = 0.0;
        int high_idx = -1;
        for (int i = 1; i < threshold - 1; i++) {
            npy_intp b_len = buckets[i].end - buckets[i].start;
            if (b_len > 1 && sse[i] > max_sse) {
                max_sse = sse[i];
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

            double sum = sse[i] + sse[i + 1];
            if (sum < min_sum) {
                min_sum = sum;
                low_idx = i;
            }
        }

        // If we can't find buckets to merge, optimization is complete
        if (low_idx < 0)
            break;

        // Step 4. Split the highest SSE bucket perfectly in half.
        npy_intp total_len = buckets[high_idx].end - buckets[high_idx].start;
        npy_intp half = (npy_intp)ceil((double)total_len / 2.0);
        npy_intp orig_end = buckets[high_idx].end;

        // Shift all buckets to the right to make room for the newly split
        // bucket
        for (int i = threshold; i > high_idx + 1; i--) {
            buckets[i] = buckets[i - 1];
        }

        // Assign boundaries for the two newly split halves
        buckets[high_idx].end = buckets[high_idx].start + half;
        buckets[high_idx + 1].start = buckets[high_idx].end;
        buckets[high_idx + 1].end = orig_end;

        // Step 5. Merge the pair selected before the split. A split to its
        // left shifted the pair one position to the right.
        if (low_idx > high_idx) {
            low_idx++;
        }

        buckets[low_idx].end = buckets[low_idx + 1].end;
        for (int i = low_idx + 1; i < threshold; i++) {
            buckets[i] = buckets[i + 1];
        }
    }

    // ==========================================
    // LTTB Point Selection Phase
    // ==========================================
    // Now that buckets are optimally sized, select the best point from each
    // bucket.

    // Always select the exact first point
    out_x[0] = points[0];
    out_y[0] = points[1];
    double last_x = out_x[0];
    double last_y = out_y[0];

    // Iterate over the middle buckets to pick the point that forms the largest
    // triangle
    for (int i = 1; i < threshold - 1; i++) {

        // 1. Calculate the average point (centroid) of the *next* bucket
        npy_intp next_start = buckets[i + 1].start;
        npy_intp next_end = buckets[i + 1].end;
        npy_intp next_count = next_end - next_start;

        double avg_next_x = 0.0, avg_next_y = 0.0;
        for (npy_intp k = next_start; k < next_end; k++) {
            avg_next_x += points[k * 2];
            avg_next_y += points[k * 2 + 1];
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
            double px = points[j * 2];
            double py = points[j * 2 + 1];

            double area = calc_triangle_area(last_x, last_y, px, py, avg_next_x,
                                             avg_next_y);
            if (area > max_area) {
                max_area = area;
                best_idx = j;
            }
        }

        // Store the selected best point and update 'last_x / last_y' for the
        // next iteration
        out_x[i] = points[best_idx * 2];
        out_y[i] = points[best_idx * 2 + 1];
        last_x = out_x[i];
        last_y = out_y[i];
    }

    // Always select the exact last point
    npy_intp last_pt_idx = (num_points - 1) * 2;
    out_x[threshold - 1] = points[last_pt_idx];
    out_y[threshold - 1] = points[last_pt_idx + 1];

    // Clean up allocated heap memory
    free(buckets);
    free(sse);
}


static PyObject *largest_triangle_dynamic(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    PyArrayObject *x = NULL, *y = NULL;
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

    // Convert input objects to contiguous C-aligned double arrays (NPY_DOUBLE)
    // NPY_ARRAY_IN_ARRAY ensures we get read-only, well-behaved contiguous
    // memory
    x = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                          NPY_ARRAY_IN_ARRAY);
    y = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
                                          NPY_ARRAY_IN_ARRAY);

    // Check if conversion failed
    if (!x || !y)
        goto fail;

    // Ensure they are 1-Dimensional and equal length
    if (PyArray_NDIM(x) != 1 || PyArray_NDIM(y) != 1 ||
        !PyArray_SAMESHAPE(x, y)) {
        PyErr_SetString(PyExc_ValueError,
                        "x and y must be 1D with identical shape.");
        goto fail;
    }

    npy_intp len_points = PyArray_DIM(x, 0);

    // If the data is already smaller than the threshold, just return the data
    // untouched
    if (threshold >= len_points) {
        PyObject *result = PyTuple_Pack(2, x, y);
        Py_DECREF(x);
        Py_DECREF(y);
        return result;
    }

    // Allocate memory for interleaved points [x1, y1, x2, y2, ...]
    // We interleave to maximize CPU cache locality during the math-heavy worker
    // phase
    npy_intp points_dims[2] = {len_points, 2};
    PyArrayObject *points =
        (PyArrayObject *)PyArray_SimpleNew(2, points_dims, NPY_DOUBLE);

    if (!points)
        goto fail;

    double *points_data = (double *)PyArray_DATA(points);
    double *x_raw = (double *)PyArray_DATA(x);
    double *y_raw = (double *)PyArray_DATA(y);

    // Perform the interleaving copy
    for (npy_intp i = 0; i < len_points; i++) {
        points_data[i * 2] = x_raw[i];
        points_data[i * 2 + 1] = y_raw[i];
    }

    // Allocate the output arrays
    npy_intp out_dim = threshold;
    PyObject *out_x_arr = PyArray_ZEROS(1, &out_dim, NPY_DOUBLE, 0);
    PyObject *out_y_arr = PyArray_ZEROS(1, &out_dim, NPY_DOUBLE, 0);

    if (!out_x_arr || !out_y_arr) {
        Py_XDECREF(out_x_arr);
        Py_XDECREF(out_y_arr);
        Py_DECREF(points);
        goto fail;
    }

    // Get raw C pointers to the newly allocated output arrays
    double *out_x = (double *)PyArray_DATA((PyArrayObject *)out_x_arr);
    double *out_y = (double *)PyArray_DATA((PyArrayObject *)out_y_arr);

    Py_BEGIN_ALLOW_THREADS

        run_ltd(points_data, len_points, threshold, out_x, out_y);

    Py_END_ALLOW_THREADS

        Py_DECREF(points);
    Py_DECREF(x);
    Py_DECREF(y);

    // Pack the output X and Y arrays into a Python tuple and return it
    PyObject *result = PyTuple_Pack(2, out_x_arr, out_y_arr);
    Py_DECREF(out_x_arr);
    Py_DECREF(out_y_arr);

    return result;

fail:
    Py_XDECREF(x);
    Py_XDECREF(y);
    return NULL;
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
