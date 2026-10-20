#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#include "matrix.h"

static void test_zeros_matrix(void)
{
    matrix_t m = zeros_matrix(3, 4);

    assert(m.data != NULL);
    assert(m.rows == 3);
    assert(m.cols == 4);
    assert(m.size == 12);

    for (size_t i = 0; i < m.size; ++i)
        assert(m.data[i] == 0.0);

    free_matrix(&m);
}

static void test_random_matrix(void)
{
    matrix_t m = random_matrix(3, 4);

    assert(m.data != NULL);
    assert(m.rows == 3);
    assert(m.cols == 4);
    assert(m.size == 12);

    for (size_t i = 0; i < m.size; ++i)
        assert(m.data[i] >= -0.1 && m.data[i] <= 0.1);

    free_matrix(&m);
}

static void test_invalid_dimensions(void)
{
    matrix_t zero_rows = zeros_matrix(0, 4);
    matrix_t zero_cols = zeros_matrix(3, 0);
    matrix_t overflow = zeros_matrix(SIZE_MAX, 2);

    assert(zero_rows.data == NULL);
    assert(zero_cols.data == NULL);
    assert(overflow.data == NULL);
}

static void test_free_matrix(void)
{
    matrix_t m = zeros_matrix(2, 2);
    assert(m.data != NULL);

    free_matrix(&m);

    assert(m.data == NULL);
    assert(m.rows == 0);
    assert(m.cols == 0);
    assert(m.size == 0);

    /* Freeing an already-cleared matrix should be safe. */
    free_matrix(&m);
}

int main(void)
{
    test_zeros_matrix();
    test_random_matrix();
    test_invalid_dimensions();
    test_free_matrix();

    puts("All matrix tests passed.");
    return 0;
}
