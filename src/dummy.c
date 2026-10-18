#include "dummy.h"

#include <stdlib.h>
#include <stdio.h>

//******************************************************************************
// TESTS THE NETWORK 
//******************************************************************************

// TODO: Only for testing 
matrix_t create_toy_data() {
    matrix_t m;
    m.rows=6;
    m.cols=6;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {
        1,2,3,3,2,1, 
        5,5,4,5,4,5, 
        4,5,4,5,5,4, 
        9,8,9,7,8,9, 
        3,2,3,1,2,2,
        7,8,7,7,8,9,
    };
    for (size_t i=0; i<m.rows*m.cols; ++i)
        m.data[i] = d[i];
    return m;
}

// TODO: Only for testing 
matrix_t create_binary_labels() {
    matrix_t m;
    m.rows=6;
    m.cols=1;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {0, 1, 0, 1, 0, 1};
    for (size_t i=0; i<m.rows*m.cols; ++i)
        m.data[i] = d[i];
    return m;
}

matrix_t create_multiclass_labels() {
    matrix_t m;
    m.rows=6;
    m.cols=3;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {
        1, 0, 0,
        0, 1, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 0, 1};
    for (size_t i=0; i<m.rows*m.cols; ++i)
        m.data[i] = d[i];
    return m;
}

void binary_test() {
    printf("BINARY\n");
    // Create matrix full of data 
    matrix_t x = create_toy_data();
    matrix_t y = create_binary_labels();

    int features = x.cols;

    network_t nn = neural_net(features, 0.1, 300, BINARY_CROSS_ENTROPY);
    add_layer(&nn, 10, relu);
    add_layer(&nn, 1, sigmoid);

    train(&nn, x, y, 1e-8, 1);
    binary_evaluator(&nn, x, y);

    // TODO: Consider an easier way for a user to construct predictions
    double new_x[] = {2,1,1,2,3,1};
    matrix_t m = {.rows = 1, .cols = features, .data = new_x};
    matrix_t preds = {0};
    predict(&nn, m, &preds);

    size_t best = 0;
    for (size_t i=0; i<nn.output_classes; ++i)
        if (preds.data[i] > best)
            best = i;
    printf("For 2,1,1,2,3,1, network predicted: %d\n", preds.data[best] > 0.5 ? 1 : 0);

    cleanup(&nn);
    free_matrix(&x);
    free_matrix(&y);
}

void multiclass_test() {
    printf("MULTICLASS\n");
    // Create matrix full of data 
    matrix_t x = create_toy_data();
    matrix_t y = create_multiclass_labels();
    size_t features = x.cols;

    network_t nn = neural_net(features, 0.1, 300, CATEGORICAL_CROSS_ENTROPY);
    add_layer(&nn, 10, relu);
    add_layer(&nn, 10, relu);
    add_layer(&nn, 3, softmax);

    train(&nn, x, y, 1e-8, 1);
    evaluate(&nn, x, y);

    // TODO: Consider an easier way for a user to construct predictions
    double new_x[] = {5,5,5,6,7,4};
    matrix_t m = {.rows = 1, .cols = features, .data = new_x};
    matrix_t preds = {0};
    predict(&nn, m, &preds);

    size_t best = 0;
    for (size_t i=0; i<m.rows * nn.output_classes; ++i) {
        if (preds.data[i] > preds.data[best])
            best = i;
        printf("preds[%zu]: %f, best idx: %zu | val %f\n", i, preds.data[i], best, preds.data[best]);
    }
    printf("For 5,5,5,6,7,4 network predicted: %zu\n", best);

    cleanup(&nn);
    free_matrix(&x);
    free_matrix(&y);
    printf("Script end\n");
}

