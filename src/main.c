#include "nn.h"
#include "mnist.h"
#include "dummy.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    // binary_test();
    // multiclass_test();
    char mnist_images[] = "./data/mnist/train-images-idx3-ubyte";
    char mnist_labels[] = "./data/mnist/train-labels-idx1-ubyte";
    char test_images[] = "./data/mnist/t10k-images-idx3-ubyte";
    char test_labels[] = "./data/mnist/t10k-labels-idx1-ubyte";
    
    matrix_t x_train = load_mnist_images(mnist_images);
    matrix_t y_train = load_mnist_labels(mnist_labels);

    matrix_t x_test = load_mnist_images(test_images);
    matrix_t y_test = load_mnist_labels(test_labels);
    
    printf("x_train Shape: %zu x %zu\n", x_train.rows, x_train.cols);
    printf("y_train Shape: %zu x %zu\n", y_train.rows, y_train.cols);
    printf("First pixel: %f\n", x_train.data[0]);

    if (x_train.rows == y_train.rows) 
        printf("x.rows == y.rows\n");

    // mnist_print_sample(&x_train, &y_train, 1990);
    // Create a subset to test with 
    int n = 100;
    matrix_t x_small = {
        .rows = n,
        .cols = x_train.cols,
        .size = n * x_train.cols,
        .data = malloc(n * x_train.cols * sizeof(double))
    };
    matrix_t y_small = {
        .rows = n,
        .cols = y_train.cols,
        .size = n * y_train.cols,
        .data = malloc(n * y_train.cols * sizeof(double))
    };
    for (size_t i=0; i<x_small.size; ++i)
        x_small.data[i] = x_train.data[i];
    for (size_t i=0; i<y_small.size; ++i)
        y_small.data[i] = y_train.data[i];


    network_t nn;
    neural_net(&nn, x_small.cols, 0.2, 50, categorical_cross_entropy);
    set_seed(&nn, 69);
    add_layer(&nn, 32, relu);
    add_layer(&nn, y_small.cols, softmax);

    train(&nn, x_small, y_small);
    printf("Train set ");
    evaluate(&nn, x_small, y_small);
    printf("Test set ");
    evaluate(&nn, x_test, y_test);

    free(x_train.data);
    free(y_train.data);
    free(x_small.data);
    free(y_small.data);
    return 0;
}
