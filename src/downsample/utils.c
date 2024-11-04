#include "utils.h"
#include <math.h>

double calculate_triangle_area(const double *point1, const double *point2,
                               const double *point3) {
    return fabs((point1[0] * (point2[1] - point3[1]) +
                 point2[0] * (point3[1] - point1[1]) +
                 point3[0] * (point1[1] - point2[1])) /
                2.0);
}
