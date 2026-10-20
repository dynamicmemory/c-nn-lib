#include "matrix.h"

#include <stdint.h>
#include <stdlib.h>

matrix_t random_matrix(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols)
        return (matrix_t){0}; 

    size_t size = rows * cols;
    if (size > SIZE_MAX / sizeof(double))
        return (matrix_t){0}; 

    matrix_t w = {
        .rows = rows, 
        .cols = cols,
        .size = size,
        .data = malloc(size * sizeof(double))
    };

    if (w.data == NULL) 
        return (matrix_t){0}; 

    for (size_t i=0; i<w.size; ++i)
        w.data[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    return w;
}

matrix_t zeros_matrix(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols)
        return (matrix_t){0}; 

    size_t size = rows * cols;
    if (size > SIZE_MAX / sizeof(double))
        return (matrix_t){0}; 

    matrix_t w = {
        .rows = rows, 
        .cols = cols,
        .size = size,
        .data = calloc(size, sizeof(double))
    };

    if (w.data == NULL) 
        return (matrix_t){0}; 

    return w;
}

void free_matrix(matrix_t *m) {
    if (m == NULL) 
        return;

    free(m->data);
    *m = (matrix_t){0};
}
