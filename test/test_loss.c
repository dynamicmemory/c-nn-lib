#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "loss.h"

#define EPSILON 1e-9


static void assert_close(double actual, double expected)
{
    if (fabs(actual - expected) >= EPSILON) {
        fprintf(stderr,
                "Expected: %.12f, got: %.12f, difference: %.12f\n",
                expected, actual, fabs(actual - expected));
        assert(fabs(actual - expected) < EPSILON);
    }
}

static layer_t make_output_layer(size_t rows, size_t cols,
                                 const double *values)
{
    layer_t layer = {0};

    layer.neurons = cols;
    layer.a.rows = rows;
    layer.a.cols = cols;
    layer.a.size = rows * cols;
    layer.a.data = (double *)values;


    return layer;
}

static matrix_t make_labels(size_t rows, size_t cols,
                            const double *values)
{
    matrix_t y = {0};

    y.rows = rows;
    y.cols = cols;
    y.size = rows * cols;
    y.data = (double *)values;

    return y;
}

static void test_binary_cross_entropy(void)
{
    const double predictions[] = {0.8, 0.25};
    const double labels[] = {1.0, 0.0};

    layer_t layer = make_output_layer(2, 1, predictions);
    matrix_t y = make_labels(2, 1, labels);

    const double expected =
        -(log(0.8) + log(1.0 - 0.25)) / 2.0;

    assert_close(binary_cross_entropy(layer, y), expected);
}

static void test_categorical_cross_entropy(void)
{
    const double predictions[] = {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3
    };

    const double labels[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    };

    layer_t layer = make_output_layer(2, 3, predictions);
    matrix_t y = make_labels(2, 3, labels);

    const double expected = -(log(0.7) + log(0.6)) / 2.0;

    assert_close(categorical_cross_entropy(layer, y), expected);
}

static void test_binary_cross_entropy_gradient(void)
{
    const double predictions[] = {0.8, 0.25};
    const double labels[] = {1.0, 0.0};

    layer_t layer = make_output_layer(2, 1, predictions);
    matrix_t y = make_labels(2, 1, labels);

    matrix_t dA = {
        .rows = 2,
        .cols = 1,
        .size = 2,
        .data = (double[2]){0.0, 0.0}
    };

    binary_cross_entropy_gradient(layer, y, &dA);

    assert_close(dA.data[0], -1.0 / 0.8);
    assert_close(dA.data[1], 1.0 / (1.0 - 0.25));
}

static void test_categorical_cross_entropy_gradient(void)
{
    const double predictions[] = {
        0.7, 0.2, 0.1,
        0.1, 0.6, 0.3
    };

    const double labels[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0
    };

    layer_t layer = make_output_layer(2, 3, predictions);
    matrix_t y = make_labels(2, 3, labels);

    double gradient_data[6] = {0};

    matrix_t dA = {
        .rows = 2,
        .cols = 3,
        .size = 6,
        .data = gradient_data
    };

    categorical_cross_entropy_gradient(layer, y, &dA);

    assert_close(dA.data[0], -1.0 / 0.7);
    assert_close(dA.data[1], 0.0);
    assert_close(dA.data[2], 0.0);

    assert_close(dA.data[3], 0.0);
    assert_close(dA.data[4], -1.0 / 0.6);
    assert_close(dA.data[5], 0.0);
}

static void test_loss_configurations(void)
{
    assert(BINARY_CROSS_ENTROPY.value == binary_cross_entropy);
    assert(BINARY_CROSS_ENTROPY.gradient ==
           binary_cross_entropy_gradient);

    assert(CATEGORICAL_CROSS_ENTROPY.value ==
           categorical_cross_entropy);
    assert(CATEGORICAL_CROSS_ENTROPY.gradient ==
           categorical_cross_entropy_gradient);
}

int main(void) {
    puts("Testing BCE...");
    test_binary_cross_entropy();

    puts("Testing CCE...");
    test_categorical_cross_entropy();

    puts("Testing BCE gradient...");
    test_binary_cross_entropy_gradient();

    puts("Testing CCE gradient...");
    test_categorical_cross_entropy_gradient();

    puts("Testing loss configurations...");
    test_loss_configurations();

    puts("All loss tests passed.");
    return 0;
}
