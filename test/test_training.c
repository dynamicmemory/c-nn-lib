#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "network.h"
#include "training.h"
#include "matrix.h"
#include "activation.h"
#include "loss.h"
#include "nnlib_internal.h"

#define EPSILON 1e-9

static void assert_close(double actual, double expected)
{
    if (fabs(actual - expected) > EPSILON) {
        fprintf(stderr, "Expected %.12f, got %.12f\n", expected, actual);
        assert(0);
    }
}

static matrix_t make_matrix(size_t rows, size_t cols, const double *values)
{
    matrix_t m = {
        .rows = rows,
        .cols = cols,
        .size = rows * cols,
        .data = malloc(rows * cols * sizeof(double))
    };

    assert(m.data != NULL);

    for (size_t i = 0; i < m.size; ++i)
        m.data[i] = values[i];

    return m;
}

static void test_training_succeeds(void)
{
    const double x_values[] = {0, 1, 1, 0};
    const double y_values[] = {0, 1, 1, 0};

    matrix_t x = make_matrix(4, 1, x_values);
    matrix_t y = make_matrix(4, 1, y_values);

    network_t nn = neural_net(1, 0.1, 10, BINARY_CROSS_ENTROPY);
    assert(add_layer(&nn, 1, sigmoid) == 0);

    assert(train(&nn, x, y, 0.0, 0) == 0);

    network_cleanup(&nn);
    free_matrix(&x);
    free_matrix(&y);
}

static void test_weights_change(void)
{
    const double x_values[] = {0, 1, 2, 3};
    const double y_values[] = {0, 0, 1, 1};

    matrix_t x = make_matrix(4, 1, x_values);
    matrix_t y = make_matrix(4, 1, y_values);

    network_t nn = neural_net(1, 0.1, 1, BINARY_CROSS_ENTROPY);
    assert(add_layer(&nn, 1, sigmoid) == 0);

    double initial_weight = nn.layers[0].W.data[0];
    double initial_bias = nn.layers[0].bias.data[0];

    assert(train(&nn, x, y, 0.0, 0) == 0);

    assert(nn.layers[0].W.data[0] != initial_weight ||
           nn.layers[0].bias.data[0] != initial_bias);

    network_cleanup(&nn);
    free_matrix(&x);
    free_matrix(&y);
}

static void test_loss_decreases(void)
{
    const double x_values[] = {0, 1, 2, 3};
    const double y_values[] = {0, 0, 1, 1};

    matrix_t x = make_matrix(4, 1, x_values);
    matrix_t y = make_matrix(4, 1, y_values);

    network_t nn = neural_net(1, 0.1, 200, BINARY_CROSS_ENTROPY);
    assert(add_layer(&nn, 1, sigmoid) == 0);

    /* Evaluate before training. */
    assert(allocate_activations(&nn.layers[0], x) == 0);
    assert(forward(&nn.layers[0], x) == 0);

    double initial_loss = nn.loss.value(nn.layers[0], y);
    free_activations(&nn);

    assert(train(&nn, x, y, 0.0, 0) == 0);

    assert(allocate_activations(&nn.layers[0], x) == 0);
    assert(forward(&nn.layers[0], x) == 0);

    double final_loss = nn.loss.value(nn.layers[0], y);
    assert(final_loss < initial_loss);

    free_activations(&nn);
    network_cleanup(&nn);
    free_matrix(&x);
    free_matrix(&y);
}

static void test_activations_are_freed(void)
{
    const double x_values[] = {0, 1};
    const double y_values[] = {0, 1};

    matrix_t x = make_matrix(2, 1, x_values);
    matrix_t y = make_matrix(2, 1, y_values);

    network_t nn = neural_net(1, 0.1, 5, BINARY_CROSS_ENTROPY);
    assert(add_layer(&nn, 1, sigmoid) == 0);

    assert(train(&nn, x, y, 0.0, 0) == 0);
    assert(nn.layers[0].a.data == NULL);
    assert(nn.layers[0].a.size == 0);

    network_cleanup(&nn);
    free_matrix(&x);
    free_matrix(&y);
}

static void test_repeatability(void)
{
    const double x_values[] = {0, 1, 2, 3};
    const double y_values[] = {0, 0, 1, 1};

    matrix_t x = make_matrix(4, 1, x_values);
    matrix_t y = make_matrix(4, 1, y_values);

    network_t nn1 = neural_net(1, 0.1, 20, BINARY_CROSS_ENTROPY);
    network_t nn2 = neural_net(1, 0.1, 20, BINARY_CROSS_ENTROPY);

    set_seed(&nn1, 123);
    assert(add_layer(&nn1, 1, sigmoid) == 0);

    set_seed(&nn2, 123);
    assert(add_layer(&nn2, 1, sigmoid) == 0);

    assert(train(&nn1, x, y, 0.0, 0) == 0);
    assert(train(&nn2, x, y, 0.0, 0) == 0);

    assert_close(nn1.layers[0].W.data[0], nn2.layers[0].W.data[0]);
    assert_close(nn1.layers[0].bias.data[0], nn2.layers[0].bias.data[0]);

    network_cleanup(&nn1);
    network_cleanup(&nn2);
    free_matrix(&x);
    free_matrix(&y);
}

int main(void) {
    puts("Testing training succeeds...");
    fflush(stdout);
    test_training_succeeds();

    puts("Testing weights change...");
    fflush(stdout);
    test_weights_change();

    puts("Testing loss decreases...");
    fflush(stdout);
    test_loss_decreases();

    puts("Testing activation cleanup...");
    fflush(stdout);
    test_activations_are_freed();

    puts("Testing repeatability...");
    fflush(stdout);
    test_repeatability();

    puts("All training tests passed.");
    return 0;
}
