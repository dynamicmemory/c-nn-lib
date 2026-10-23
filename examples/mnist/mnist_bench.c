#include "nnlib.h"
#include "mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct {
    int n_samples;
    int epochs;
    int threads;
    double learning_rate;
} args_t;

int parse_args(int argc, char *argv[], args_t *args);
double wtime(void);
int create_subset(const matrix_t *src, matrix_t *dst, size_t n);

int main(int argc, char *argv[]) {
    double full_start = wtime();

    // Check cmd line args, set defaults if none passed in
    args_t args;
    if (parse_args(argc, argv, &args) < 0)
        return EXIT_FAILURE;
    
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

    // Create a subset of data to train on (for testing different sized train sets)
    matrix_t x_subset, y_subset;
    if (create_subset(&x_train, &x_subset, args.n_samples) < 0 || 
        create_subset(&y_train, &y_subset, args.n_samples) < 0) {
        fprintf(stderr, "Unable to create a training subset\n");
        return EXIT_FAILURE;
    }

    // Setup model
    network_t nn = neural_net(features, args.learning_rate, args.epochs, CATEGORICAL_CROSS_ENTROPY);
    set_seed(&nn, 69);
    nn.threads = args.threads;

    int l1 = add_layer(&nn, 32, relu);
    int l2 = add_layer(&nn, y_subset.cols, softmax);
    if (l1 < 0 || l2 < 0) {
        network_cleanup(&nn);
        return EXIT_FAILURE;
    }

    // Train 
    printf("Samples: %d\nEpochs: %d\nlr: %f\nthreads: %d\n", 
            args.n_samples, args.epochs, args.learning_rate, args.threads);
    double train_start = wtime();
    // train(&nn, x_subset, y_subset, 1e-8, 0);
    train(&nn, x_subset, y_subset, 0.0, 0);
    double train_end = wtime() - train_start;

    printf("Samples: %d\nEpochs: %d\nlr: %f\nthreads: %d\n", 
            args.n_samples, args.epochs, args.learning_rate, args.threads);

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

    double full_end = wtime() - full_start;

    printf("WORKFLOW_SECONDS=%.9f\n", full_end);
    printf("TRAIN_SECONDS=%.9f\n", train_end);

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

double wtime(void) { 
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

//TODO: extract out to parse_long() function
int parse_args(int argc, char *argv[], args_t *args) {
    if (argc != 5) {
        fprintf(stderr, "usage: <n_samples> <epochs> <threads> <learning_rate>\n");
        return -1;
    } 

    char *end;
    long result = strtol(argv[1], &end, 10);
    if (*end != '\0') { 
        fprintf(stderr, "n_samples should be an integer between 1 and training max\n");
        return -1;
    }
    args->n_samples = (int)result;

    result = strtol(argv[2], &end, 10);
    if (*end != '\0') { 
        fprintf(stderr, "epochs should be an integer or long value\n");
        return -1;
    }
    args->epochs = (int)result;

    result = strtol(argv[3], &end, 10);
    if (*end != '\0') { 
        fprintf(stderr, "threads should be an integer between 1 and system max\n");
        return -1;
    }
    args->threads = (int)result;

    printf("%s\n", argv[4]);
    double lr = strtof(argv[4], &end);
    if (*end != '\0') { 
        fprintf(stderr, "learning_rate should be an integer between 0 and 1\n");
        return -1;
    }
    args->learning_rate = lr;

    return 0;
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
