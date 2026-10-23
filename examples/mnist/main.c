/* Example script for the neural net library. Showcases the basic functionality 
 * of the library from setting up the network, adding n number of layers, 
 * training, evaluating the results, and making a prediction. 
 * 
 * This script also showcases a few of the mnist functions included in the 
 * example which can be used for testing or baselines for your own 
 * implementations
 */
#include "nnlib.h"
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Creates a subset of data from a passed in matrix, purely for this example*/
int create_subset(const matrix_t *src, matrix_t *dst, size_t n);

/* Simple workflow for making a prediction and interpretting it purely for this 
 * example only 
 */
int make_a_prediction(network_t *nn, matrix_t *x, matrix_t *y);

/* Used to time training*/
double wtime(void);

int main(void) {
    // Import the data
    char mnist_images[] = "./examples/mnist/data/train-images-idx3-ubyte";
    char mnist_labels[] = "./examples/mnist/data/train-labels-idx1-ubyte";
    char test_images[] = "./examples/mnist/data/t10k-images-idx3-ubyte";
    char test_labels[] = "./examples/mnist/data/t10k-labels-idx1-ubyte";
    
    matrix_t x_train = load_mnist_images(mnist_images);
    matrix_t y_train = load_mnist_labels(mnist_labels);
    matrix_t x_test = load_mnist_images(test_images);
    matrix_t y_test = load_mnist_labels(test_labels);
    
    size_t features = x_train.cols;

    // Network variables 
    int n_samples = 60000;
    int epochs = 100;
    int threads = 6;
    double lr = 0.2;

    // Create a subset of data to train on (for testing different sized train sets)
    matrix_t x_subset, y_subset;
    if (create_subset(&x_train, &x_subset, n_samples) < 0 || 
        create_subset(&y_train, &y_subset, n_samples) < 0) {
        fprintf(stderr, "Unable to create a training subset\n");
        return EXIT_FAILURE;
    }

    // Setup and build the network
    network_t nn = neural_net(features, lr, epochs, CATEGORICAL_CROSS_ENTROPY);
    set_seed(&nn, 69);
    nn.threads = threads;

    int l1 = add_layer(&nn, 32, relu);
    int l2 = add_layer(&nn, y_subset.cols, softmax);
    if (l1 < 0 || l2 < 0) {
        network_cleanup(&nn);
        return EXIT_FAILURE;
    }

    // Train the model 
    printf("Samples: %d\nEpochs: %d\nlr: %f\nthreads: %d\n", 
            n_samples, epochs, lr, threads);
    double start_clock = wtime();
    train(&nn, x_subset, y_subset, 1e-8, 1);
    double training_time = wtime() - start_clock;
    printf("Samples: %d\nEpochs: %d\nlr: %f\nthreads: %d\nTraining Time: %f", 
            n_samples, epochs, lr, threads, training_time);

    // Evaluate 
    printf("Train set ");
    evaluate_t eval;
    if (evaluate(&nn, x_subset, y_subset, &eval, 0.5) == 0) {
        print_confusion_matrix(eval.confusion);
        print_metrics(&eval);
        free_evaluate(&eval);
        printf("\n");
    }

    printf("Test set ");
    if (evaluate(&nn, x_test, y_test, &eval, 0.5) == 0) {
        print_confusion_matrix(eval.confusion);
        print_metrics(&eval);
        free_evaluate(&eval);
        printf("\n");
    }

    // Make a prediction
    make_a_prediction(&nn, &x_test, &y_test);

    // Clean up
    free_matrix(&x_train);
    free_matrix(&y_train);
    free_matrix(&x_test);
    free_matrix(&y_test);
    free_matrix(&x_subset);
    free_matrix(&y_subset);

    network_cleanup(&nn);
    return EXIT_SUCCESS;
}

int create_subset(const matrix_t *src, matrix_t *dst, size_t n) {
    if (src->rows < n)
        return -1;

    dst->rows = n;
    dst->cols = src->cols;
    dst->size = n * src->cols;
    dst->data = malloc(n*src->cols * sizeof(double));

    if (dst->data == NULL) 
        return -1;

    for (size_t i=0; i<dst->size; ++i)
        dst->data[i] = src->data[i];

    return 0;
}

int make_a_prediction(network_t *nn, matrix_t *x, matrix_t *y) {

    matrix_t sample, preds;
    if(create_subset(x, &sample, 1) < 0) {
        fprintf(stderr, "Unable to create preds set\n");
        return -1;
    }
    predict(nn, sample, &preds);
    int predicted = 0;
    printf("Models class probabilities: \n");
    for (size_t col=0; col<nn->output_classes; ++col) {
        printf("Class: %zu - Probability: %f\n", col, preds.data[col]);
        if (preds.data[col] > preds.data[predicted])
            predicted = col;
    }

    mnist_print_sample(&sample, y, 0);
    printf("The model predicted: %d\n", predicted);

    free_matrix(&sample);
    free_matrix(&preds);
    return 0;
}

double wtime(void) { 
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
