#include "llnnic.h"
#include "mnist.h"
#include "dummy.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
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
    
    // Create a subset to test with 
    int n = 60000;
    int epochs = 5000;
    double lr = 0.2;
    size_t threads = 12;

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


    printf("Samples: %d\nEpochs: %d\nlr: %f\n", n, epochs, lr);
    network_t nn = neural_net(x_small.cols, lr, epochs, CATEGORICAL_CROSS_ENTROPY);
    set_seed(&nn, 69);

    nn.threads = threads;

    int l1 = add_layer(&nn, 32, relu);
    int l2 = add_layer(&nn, y_small.cols, softmax);
    if (l1 < 0 || l2 < 0) {
        network_cleanup(&nn);
        return 1;
    }

    train(&nn, x_small, y_small, 1e-8, 1);

    printf("Samples: %d\nEpochs: %d\nlr: %fthreads: %zu\n", n, epochs, lr, threads);

    printf("Train set ");
    evaluate_t eval;
    if (evaluate(&nn, x_small, y_small, &eval, 0.5) == 0) {
        print_confusion_matrix(eval.confusion);
        print_metrics(&eval);
        free_evaluate(&eval);
    }
    printf("Test set ");
    if (evaluate(&nn, x_test, y_test, &eval, 0.5) == 0) {
        print_confusion_matrix(eval.confusion);
        print_metrics(&eval);
        free_evaluate(&eval);
    }

    free(x_train.data);
    free(y_train.data);
    free(x_test.data);
    free(y_test.data);
    free(x_small.data);
    free(y_small.data);
    network_cleanup(&nn);
    return 0;
}
