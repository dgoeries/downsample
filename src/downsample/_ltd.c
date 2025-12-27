#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#define PY_SSIZE_T_CLEAN
#include <numpy/arrayobject.h>

#define __DOUBLE_SIZE__ sizeof(double)
#include "utils.h"

static PyObject *calculate_average_point(PyArrayObject *bucket) {
    npy_intp num_points = PyArray_DIM(bucket, 0);
    npy_intp dim = PyArray_DIM(bucket, 1);

    npy_intp dims[1] = {dim};
    PyObject *avg_point = PyArray_ZEROS(1, dims, NPY_DOUBLE, 0);
    double *avg_data = (double *)PyArray_DATA((PyArrayObject *)avg_point);

    double *bucket_data = (double *)PyArray_DATA(bucket);
    for (npy_intp i = 0; i < num_points; i++) {
        for (npy_intp d = 0; d < dim; d++) {
            avg_data[d] += bucket_data[i * dim + d];
        }
    }
    // calculate the average for each dimension
    for (npy_intp d = 0; d < dim; d++) {
        avg_data[d] /= (double)num_points;
    }
    return avg_point;
}


static PyObject *split_bucket_at(PyObject *buckets_list, int index) {
    Py_ssize_t bucket_count = PyList_Size(buckets_list);
    if (index < 0 || index >= bucket_count) {
        PyErr_SetString(PyExc_IndexError, "Index out of range for split");
        return NULL;
    }

    // Get the bucket of interest
    PyArrayObject *bucket =
        (PyArrayObject *)PyList_GetItem(buckets_list, index);

    npy_intp bucket_size = PyArray_DIM(bucket, 0);
    npy_intp dim = PyArray_DIM(bucket, 1);

    // calculate split sizes
    npy_intp bucket_a_length = (npy_intp)ceil((double)bucket_size / 2.0);
    npy_intp bucket_b_length = bucket_size - bucket_a_length;
    npy_intp dims_a[2] = {bucket_a_length, dim};
    npy_intp dims_b[2] = {bucket_b_length, dim};

    // get views
    double *bucket_a_data = (double *)PyArray_DATA(bucket);
    double *bucket_b_data =
        (double *)PyArray_GETPTR2(bucket, bucket_a_length, 0);

    PyObject *bucket_a =
        PyArray_SimpleNewFromData(2, dims_a, NPY_DOUBLE, bucket_a_data);
    PyObject *bucket_b =
        PyArray_SimpleNewFromData(2, dims_b, NPY_DOUBLE, bucket_b_data);

    // Ensure bucket stays alive exactly as long as either bucket_a or
    // bucket_b exists.
    Py_INCREF(bucket);
    PyArray_SetBaseObject((PyArrayObject *)bucket_a, (PyObject *)bucket);
    Py_INCREF(bucket);
    PyArray_SetBaseObject((PyArrayObject *)bucket_b, (PyObject *)bucket);

    PyObject *result_buckets = PyList_New(bucket_count + 1);
    for (Py_ssize_t i = 0; i < bucket_count; i++) {
        if (i == index) {
            continue;
        }
        PyObject *item = PyList_GetItem(buckets_list, i);
        Py_INCREF(item);
        PyList_SetItem(result_buckets, (i < index) ? i : i + 1, item);
    }

    // Insert split buckets
    // PyList_SetItem steals the reference
    PyList_SetItem(result_buckets, index, bucket_a);
    PyList_SetItem(result_buckets, index + 1, bucket_b);

    return result_buckets;
}


static PyObject *merge_bucket_at(PyObject *buckets_list, int index) {
    Py_ssize_t bucket_count = PyList_Size(buckets_list);
    if (index < 0 || index >= bucket_count - 1) {
        PyErr_SetString(PyExc_IndexError, "Index out of range for merging");
        return NULL;
    }
    PyArrayObject *bucket_a =
        (PyArrayObject *)PyList_GetItem(buckets_list, index);
    PyArrayObject *bucket_b =
        (PyArrayObject *)PyList_GetItem(buckets_list, index + 1);

    PyObject *arrays_to_merge = PyList_New(2);
    Py_INCREF(bucket_a);
    Py_INCREF(bucket_b);

    PyList_SetItem(arrays_to_merge, 0, (PyObject *)bucket_a);
    PyList_SetItem(arrays_to_merge, 1, (PyObject *)bucket_b);

    PyObject *merged_bucket = PyArray_Concatenate(arrays_to_merge, 0);
    Py_DECREF(arrays_to_merge);

    PyObject *result_buckets = PyList_New(bucket_count - 1);

    for (Py_ssize_t i = 0; i < index; i++) {
        PyObject *item = PyList_GetItem(buckets_list, i);
        Py_INCREF(item);
        PyList_SetItem(result_buckets, i, item);
    }

    PyList_SetItem(result_buckets, index, merged_bucket);

    for (Py_ssize_t i = index + 2; i < bucket_count; i++) {
        PyObject *item = PyList_GetItem(buckets_list, i);
        Py_INCREF(item);
        PyList_SetItem(result_buckets, i - 1, item);
    }

    // No DECREF of buckets_list here (borrowed ref)
    return result_buckets;
}

static PyObject *LTTB_for_buckets(PyObject *buckets_list) {
    Py_ssize_t bucket_count = PyList_Size(buckets_list);
    npy_intp num_points = bucket_count;
    PyObject *x_array = PyArray_ZEROS(1, &num_points, NPY_DOUBLE, 0);
    PyObject *y_array = PyArray_ZEROS(1, &num_points, NPY_DOUBLE, 0);

    double *x_data = (double *)PyArray_DATA((PyArrayObject *)x_array);
    double *y_data = (double *)PyArray_DATA((PyArrayObject *)y_array);

    PyArrayObject *first_bucket =
        (PyArrayObject *)PyList_GetItem(buckets_list, 0);

    double *first_point_data = (double *)PyArray_GETPTR2(first_bucket, 0, 0);
    x_data[0] = first_point_data[0];
    y_data[0] = first_point_data[1];
    double *last_selected_data = first_point_data;

    for (Py_ssize_t i = 1; i < bucket_count - 1; i++) {
        PyArrayObject *bucket =
            (PyArrayObject *)PyList_GetItem(buckets_list, i);
        PyArrayObject *next_bucket =
            (PyArrayObject *)PyList_GetItem(buckets_list, i + 1);

        PyObject *average_point_obj = calculate_average_point(next_bucket);
        PyArrayObject *average_point = (PyArrayObject *)average_point_obj;
        double *average_point_data = (double *)PyArray_DATA(average_point);

        double max_area = -1.0;
        npy_intp max_area_index = -1;
        npy_intp bucket_size = PyArray_DIM(bucket, 0);

        for (npy_intp j = 0; j < bucket_size; j++) {
            double *point_data = (double *)PyArray_GETPTR2(bucket, j, 0);
            double area = calculate_triangle_area(
                last_selected_data, point_data, average_point_data);
            if (area > max_area) {
                max_area = area;
                max_area_index = j;
            }
        }
        double *selected_point_data =
            (double *)PyArray_GETPTR2(bucket, max_area_index, 0);
        x_data[i] = selected_point_data[0];
        y_data[i] = selected_point_data[1];
        last_selected_data = selected_point_data;
        Py_DECREF(average_point);
    }

    PyArrayObject *last_bucket =
        (PyArrayObject *)PyList_GetItem(buckets_list, bucket_count - 1);
    double *last_point_data = (double *)PyArray_GETPTR2(last_bucket, 0, 0);
    x_data[bucket_count - 1] = last_point_data[0];
    y_data[bucket_count - 1] = last_point_data[1];

    PyObject *result = PyTuple_Pack(2, x_array, y_array);
    Py_DECREF(x_array);
    Py_DECREF(y_array);

    return result;
}


static double calculate_sse_for_bucket(PyArrayObject *bucket) {
    npy_intp num_points = PyArray_DIM(bucket, 0);
    double *data = (double *)PyArray_DATA(bucket);

    double sum_x = 0.0, sum_y = 0.0;
    double a_numerator = 0.0;
    double a_denominator = 0.0;

    for (npy_intp i = 0; i < num_points; i++) {
        sum_x += data[i * 2];
        sum_y += data[i * 2 + 1];
    }
    double avg_x = sum_x / (double)num_points;
    double avg_y = sum_y / (double)num_points;

    for (npy_intp i = 0; i < num_points; i++) {
        double deviation_x = data[i * 2] - avg_x;
        double deviation_y = data[i * 2 + 1] - avg_y;
        a_numerator += deviation_x * deviation_y;
        a_denominator += deviation_x * deviation_x;
    }
    double a, b;
    if (a_denominator == 0.0) {
        a = (a_numerator > 0) ? INFINITY : -INFINITY;
        b = avg_y;
    } else {
        a = a_numerator / a_denominator;
        b = avg_y - a * avg_x;
    }
    double sse = 0.0;
    for (npy_intp i = 0; i < num_points; i++) {
        double x = data[i * 2];
        double y = data[i * 2 + 1];
        double standardError = y - (a * x + b);
        sse += standardError * standardError;
    }
    return sse;
}


static npy_intp find_highest_sse_bucket_index(PyObject *buckets_list,
                                              PyArrayObject *sse_array) {
    npy_intp sse_len = PyArray_DIM(sse_array, 0);
    if (sse_len < 3) {
        PyErr_SetString(
            PyExc_ValueError,
            "SSE array must contain at least three values to exclude "
            "the first and last buckets");
        return -1;
    }
    double *sse_data = (double *)PyArray_DATA(sse_array);
    double max_sse = 0.0;
    npy_intp max_sse_idx = -1;

    for (npy_intp i = 1; i < sse_len - 1; i++) {
        PyObject *bucket = PyList_GetItem(buckets_list, i);
        npy_intp bucket_dim = PyArray_DIM((PyArrayObject *)bucket, 0);
        if (bucket_dim > 1 && sse_data[i] > max_sse) {
            max_sse = sse_data[i];
            max_sse_idx = i;
        }
    }

    return max_sse_idx;
}


static npy_intp find_lowest_sse_adjacent_buckets_index(PyArrayObject *sse_array,
                                                       npy_intp ignore_index) {
    npy_intp sse_len = PyArray_DIM(sse_array, 0);
    if (sse_len < 2) {
        PyErr_SetString(
            PyExc_ValueError,
            "SSE array must contain at least two values to find adjacent sums");
        return -1;
    }
    double *sse_data = (double *)PyArray_DATA(sse_array);
    double min_sse_sum = INFINITY;
    npy_intp min_sse_index = -1;

    for (npy_intp i = 1; i < sse_len - 2; i++) {
        if (i == ignore_index || i + 1 == ignore_index) {
            continue;
        }
        double adjacent_sum = sse_data[i] + sse_data[i + 1];
        if (adjacent_sum < min_sse_sum) {
            min_sse_sum = adjacent_sum;
            min_sse_index = i;
        }
    }
    return min_sse_index;
}


static PyObject *calculate_sse_for_buckets(PyObject *buckets_list) {
    Py_ssize_t num_buckets = PyList_Size(buckets_list);
    if (num_buckets < 3) {
        PyErr_SetString(PyExc_ValueError,
                        "Not enough buckets to calculate SSE");
        return NULL;
    }
    npy_intp sse_len = num_buckets;
    PyObject *sse_array =
        PyArray_Zeros(1, &sse_len, PyArray_DescrFromType(NPY_DOUBLE), 0);

    double *sse_data = (double *)PyArray_DATA((PyArrayObject *)sse_array);

    for (Py_ssize_t i = 1; i < num_buckets - 1; i++) {
        PyArrayObject *prev_bucket =
            (PyArrayObject *)PyList_GetItem(buckets_list, i - 1);
        PyArrayObject *curr_bucket =
            (PyArrayObject *)PyList_GetItem(buckets_list, i);
        PyArrayObject *next_bucket =
            (PyArrayObject *)PyList_GetItem(buckets_list, i + 1);

        npy_intp curr_rows = PyArray_DIM(curr_bucket, 0);
        npy_intp cols = PyArray_DIM(curr_bucket, 1);

        npy_intp combined_dims[2] = {curr_rows + 2, cols};
        PyArrayObject *bucket_with_adj =
            (PyArrayObject *)PyArray_SimpleNew(2, combined_dims, NPY_DOUBLE);

        double *bucket_adj_data = (double *)PyArray_DATA(bucket_with_adj);
        double *prev_last_row = (double *)PyArray_GETPTR2(
            prev_bucket, PyArray_DIM(prev_bucket, 0) - 1, 0);
        memcpy(bucket_adj_data, prev_last_row, cols * __DOUBLE_SIZE__);

        double *curr_data = (double *)PyArray_DATA(curr_bucket);
        memcpy(bucket_adj_data + cols, curr_data,
               curr_rows * cols * __DOUBLE_SIZE__);

        double *next_first_row = (double *)PyArray_GETPTR2(next_bucket, 0, 0);
        memcpy(bucket_adj_data + (curr_rows + 1) * cols, next_first_row,
               cols * __DOUBLE_SIZE__);

        sse_data[i] = calculate_sse_for_bucket(bucket_with_adj);
        Py_DECREF(bucket_with_adj);
    }

    return sse_array;
}

static PyObject *ltd_for_buckets(PyObject *buckets_list) {
    Py_INCREF(buckets_list);

    Py_ssize_t num_buckets = PyList_Size(buckets_list);
    int threshold = (int)num_buckets;
    int num_iterations = ((int)num_buckets * 10) / threshold;

    for (int i = 0; i < num_iterations; i++) {
        PyObject *sse_for_buckets = calculate_sse_for_buckets(buckets_list);

        npy_intp highest_sse_bucket_index = find_highest_sse_bucket_index(
            buckets_list, (PyArrayObject *)sse_for_buckets);

        if (highest_sse_bucket_index < 0) {
            Py_DECREF(sse_for_buckets);
            break;
        }

        npy_intp lowest_sse_adjacent_bucket_index =
            find_lowest_sse_adjacent_buckets_index(
                (PyArrayObject *)sse_for_buckets, highest_sse_bucket_index);

        if (lowest_sse_adjacent_bucket_index < 0) {
            Py_DECREF(sse_for_buckets);
            break;
        }

        // Split
        PyObject *updated_buckets =
            split_bucket_at(buckets_list, highest_sse_bucket_index);

        Py_DECREF(buckets_list);
        buckets_list = updated_buckets;

        if (lowest_sse_adjacent_bucket_index > highest_sse_bucket_index) {
            lowest_sse_adjacent_bucket_index += 1;
        }

        // Merge
        PyObject *merged_buckets =
            merge_bucket_at(buckets_list, lowest_sse_adjacent_bucket_index);

        Py_DECREF(buckets_list);
        buckets_list = merged_buckets;

        Py_DECREF(sse_for_buckets);
    }

    PyObject *lttb_result = LTTB_for_buckets(buckets_list);

    Py_DECREF(buckets_list);

    return lttb_result;
}


static PyObject *split_into_buckets(PyArrayObject *data, int threshold) {
    npy_intp num_points = PyArray_DIM(data, 0);
    npy_intp num_cols = PyArray_DIM(data, 1);

    PyObject *buckets = PyList_New(threshold);

    // We need to add the first point as the first bucket
    npy_intp first_dims[2] = {1, num_cols};
    PyArrayObject *first_point = (PyArrayObject *)PyArray_SimpleNewFromData(
        2, first_dims, NPY_DOUBLE, PyArray_DATA(data));

    PyList_SetItem(buckets, 0, (PyObject *)first_point);

    double bucket_size = (double)(num_points - 2) / (threshold - 2);
    for (int i = 0; i < threshold - 2; i++) {
        npy_intp bucket_start_index = (npy_intp)floor(i * bucket_size) + 1;
        npy_intp bucket_end_index = (npy_intp)floor((i + 1) * bucket_size) + 1;
        npy_intp current_bucket_size = bucket_end_index - bucket_start_index;

        npy_intp bucket_dims[2] = {current_bucket_size, num_cols};

        double *bucket_start_ptr =
            (double *)PyArray_DATA(data) + bucket_start_index * num_cols;

        PyArrayObject *current_bucket =
            (PyArrayObject *)PyArray_SimpleNewFromData(
                2, bucket_dims, NPY_DOUBLE, (void *)bucket_start_ptr);

        PyList_SetItem(buckets, i + 1, (PyObject *)current_bucket);
    }

    // And the last point as the last bucket for margins
    npy_intp last_dims[2] = {1, num_cols};
    double *last_point_ptr =
        (double *)PyArray_DATA(data) + (num_points - 1) * num_cols;
    PyArrayObject *last_point = (PyArrayObject *)PyArray_SimpleNewFromData(
        2, last_dims, NPY_DOUBLE, (void *)last_point_ptr);

    PyList_SetItem(buckets, threshold - 1, (PyObject *)last_point);

    return buckets;
}


static PyObject *largest_triangle_dynamic(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj;
    PyArrayObject *x = NULL, *y = NULL;
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

    x = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_DOUBLE,
                                          NPY_ARRAY_IN_ARRAY);
    y = (PyArrayObject *)PyArray_FROM_OTF(y_obj, NPY_DOUBLE,
                                          NPY_ARRAY_IN_ARRAY);
    if (!x || !y)
        goto fail;

    if (PyArray_NDIM(x) != 1 || PyArray_NDIM(y) != 1) {
        PyErr_SetString(PyExc_ValueError, "x and y must be 1-dimensional.");
        goto fail;
    }
    if (!PyArray_SAMESHAPE(x, y)) {
        PyErr_SetString(PyExc_ValueError, "x and y must have the same shape.");
        goto fail;
    }

    npy_intp len_points = PyArray_DIM(x, 0);
    if (threshold >= len_points || len_points <= 2) {
        PyObject *result = PyTuple_Pack(2, x, y);
        Py_DECREF(x);
        Py_DECREF(y);
        return result;
    }

    npy_intp points_dims[2] = {len_points, 2};
    PyArrayObject *points =
        (PyArrayObject *)PyArray_SimpleNew(2, points_dims, NPY_DOUBLE);

    double *points_data = (double *)PyArray_DATA(points);
    for (npy_intp i = 0; i < len_points; i++) {
        points_data[i * 2] = *(double *)PyArray_GETPTR1(x, i);
        points_data[i * 2 + 1] = *(double *)PyArray_GETPTR1(y, i);
    }

    PyObject *buckets = split_into_buckets(points, threshold);
    PyObject *result = ltd_for_buckets(buckets);

    Py_DECREF(x);
    Py_DECREF(y);
    Py_DECREF(buckets);
    Py_DECREF(points);

    return result;

fail:
    Py_XDECREF(x);
    Py_XDECREF(y);
    return NULL;
}


static PyMethodDef LTDMethods[] = {{"largest_triangle_dynamic",
                                    largest_triangle_dynamic, METH_VARARGS,
                                    "Largest triangle dynamic for buckets"},
                                   {NULL, NULL, 0, NULL}};

static struct PyModuleDef LTDModule = {
    PyModuleDef_HEAD_INIT, "_ltd",
    "Module for LTD downsampling using the NumPy C API", -1, LTDMethods};

PyMODINIT_FUNC PyInit__ltd(void) {
    import_array();
    return PyModule_Create(&LTDModule);
}
