#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "evaluation.h"
#include "network.h"
#include "activation.h"
#include "loss.h"
#include "matrix.h"

#define EPSILON 1e-9

static void assert_close(double actual, double expected)
{
    if (fabs(actual - expected) > EPSILON) {
        fprintf(stderr, "Expected %.12f, got %.12f\n", expected, actual);
        assert(0);
    }
}

static matrix_t make_matrix(size_t rows, size_t cols,
                            const double *values)
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

static void test_binary_evaluation(void)
{
    const double x_values[] = {-2, -1, 1, 2};
    const double y_values[] = {0, 0, 1, 1};

    matrix_t x = make_matrix(4, 1, x_values);
    matrix_t y = make_matrix(4, 1, y_values);

    network_t nn = neural_net(1, 0.1, 10, BINARY_CROSS_ENTROPY);
    assert(add_layer(&nn, 1, sigmoid) == 0);

    /* Make predictions deterministic. */
    nn.layers[0].W.data[0] = 1.0;
    nn.layers[0].bias.data[0] = 0.0;

    evaluate_t result = {0};
    assert(evaluate(&nn, x, y, &result, 0.5) == 0);

    assert(result.confusion.rows == 2);
    assert(result.confusion.cols == 2);

    /* Rows = actual, columns = predicted. */
    assert_close(result.confusion.data[0], 2.0); /* TN */
    assert_close(result.confusion.data[1], 0.0); /* FP */
    assert_close(result.confusion.data[2], 0.0); /* FN */
    assert_close(result.confusion.data[3], 2.0); /* TP */

    assert_close(result.accuracy, 1.0);
    assert_close(result.precision, 1.0);
    assert_close(result.recall, 1.0);
    assert_close(result.f1_score, 1.0);
    assert_close(result.threshold, 0.5);
    assert(isfinite(result.loss));

    free_evaluate(&result);
    network_cleanup(&nn);
    free_matrix(&x);
    free_matrix(&y);
}

static void test_binary_prediction(void)
{
    const double x_values[] = {-1, 0, 1};
    matrix_t x = make_matrix(3, 1, x_values);

    network_t nn = neural_net(1, 0.1, 10, BINARY_CROSS_ENTROPY);
    assert(add_layer(&nn, 1, sigmoid) == 0);

    nn.layers[0].W.data[0] = 1.0;
    nn.layers[0].bias.data[0] = 0.0;

    matrix_t preds = {0};
    assert(predict(&nn, x, &preds) == 0);

    assert(preds.rows == 3);
    assert(preds.cols == 1);

    assert_close(preds.data[0], 1.0 / (1.0 + exp(1.0)));
    assert_close(preds.data[1], 0.5);
    assert_close(preds.data[2], 1.0 / (1.0 + exp(-1.0)));

    /* predict() should release temporary layer activations. */
    assert(nn.layers[0].a.data == NULL);
    assert(nn.layers[0].a.size == 0);

    free_matrix(&preds);
    network_cleanup(&nn);
    free_matrix(&x);
}

static void test_multiclass_evaluation(void)
{
    const double x_values[] = {
        2, 0,
        0, 2,
        3, 1,
        1, 3
    };

    const double y_values[] = {
        1, 0,
        0, 1,
        1, 0,
        0, 1
    };

    matrix_t x = make_matrix(4, 2, x_values);
    matrix_t y = make_matrix(4, 2, y_values);

    network_t nn = neural_net(2, 0.1, 10,
                             CATEGORICAL_CROSS_ENTROPY);
    assert(add_layer(&nn, 2, softmax) == 0);

    /* Output neuron 0 responds to feature 0; neuron 1 to feature 1. */
    nn.layers[0].W.data[0] = 1.0;
    nn.layers[0].W.data[1] = 0.0;
    nn.layers[0].W.data[2] = 0.0;
    nn.layers[0].W.data[3] = 1.0;

    for (size_t i = 0; i < nn.layers[0].bias.size; ++i)
        nn.layers[0].bias.data[i] = 0.0;

    evaluate_t result = {0};
    assert(evaluate(&nn, x, y, &result, -1.0) == 0);

    assert(result.confusion.rows == 2);
    assert(result.confusion.cols == 2);

    assert_close(result.confusion.data[0], 2.0);
    assert_close(result.confusion.data[1], 0.0);
    assert_close(result.confusion.data[2], 0.0);
    assert_close(result.confusion.data[3], 2.0);

    assert_close(result.accuracy, 1.0);
    assert_close(result.precision, 1.0);
    assert_close(result.recall, 1.0);
    assert_close(result.f1_score, 1.0);
    assert_close(result.threshold, -1.0);
    assert(isfinite(result.loss));

    free_evaluate(&result);
    network_cleanup(&nn);
    free_matrix(&x);
    free_matrix(&y);
}

static void test_free_evaluate(void)
{
    evaluate_t result = {0};
    result.confusion = zeros_matrix(2, 2);
    assert(result.confusion.data != NULL);

    result.loss = 0.5;
    result.accuracy = 0.75;

    free_evaluate(&result);

    assert(result.confusion.data == NULL);
    assert(result.confusion.rows == 0);
    assert(result.confusion.cols == 0);
    assert(result.confusion.size == 0);
    assert_close(result.loss, 0.0);
    assert_close(result.accuracy, 0.0);

    /* Repeated cleanup should be safe. */
    free_evaluate(&result);
}

int main(void)
{
    puts("Testing binary evaluation...");
    test_binary_evaluation();

    puts("Testing binary prediction...");
    test_binary_prediction();

    puts("Testing multiclass evaluation...");
    test_multiclass_evaluation();

    puts("Testing evaluation cleanup...");
    test_free_evaluate();

    puts("All evaluation tests passed.");
    return 0;
}
