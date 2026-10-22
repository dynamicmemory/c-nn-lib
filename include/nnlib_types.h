#ifndef NNLIB_TYPES_H
#define NNLIB_TYPES_H

#include <stddef.h>

/* Matrix */
/* 
 * Represents a dynamically allocated matrix of doubles, matrix elements are 
 * stored in a continguous, row-major array.
 * Element(row, col) located data[row * cols + col]
 *
 * rows: number of rows 
 * cols: number of cols 
 * size: total elements in the matrix (rows * cols)
 * data: allocated double array of elements
*/
typedef struct {
    size_t rows;
    size_t cols;
    size_t size;
    double *data;
} matrix_t;


/* Activation function pointer types */

typedef void (*activation_fn)(const double *z, double *a, size_t n);
// typedef double (*dActivation_fn)(double a);
typedef void (*dActivation_fn)(const double *a, const double *dA, double *dZ, size_t n);


/* Network layer*/

/* Represents a neural network layer, including its trainable parameters, 
 * gradients, activations, and activation functions.
 *
 * W: weight matrix (neurons x inputs)
 * bias: bias matrix (neurons x 1)
 * a: activation matrix (samles x neurons), allocated during training
 *
 * dW: weight gradient matrix (neurons x inputs)
 * dbias: bias gradient matrix (neurons x 1)
 *
 * activation: activation function for the layer 
 * d_activation: derivative of the activaiont function for the layer
 */
typedef struct {
    size_t inputs;
    size_t neurons;

    matrix_t W;
    matrix_t bias;
    matrix_t a;
    matrix_t dW;
    matrix_t dB;

    activation_fn activation;
    dActivation_fn d_activation;
} layer_t;


/* Loss functions and type */

typedef double (*loss_fn)(layer_t layer, matrix_t y);
typedef void (*loss_gradient_fn)(layer_t layer, matrix_t y, matrix_t *dA);

/*
 * Represents the loss for the network with two functions 
 * 
 * loss_fn calculates the double value scaler loss of the network 
 * loss_gradient_fn calculates the gradient of the loss with respect to the 
 * output activation.
 *
 * The loss gradient function does no average across samples, it is left to the 
 * caller to handle i.e backprop 
 */
typedef struct {
    loss_fn value;
    loss_gradient_fn gradient;
} loss_t;


/* Network type*/

/* 
 * Represents a neural network object, including its layers, training details 
 * loss function, and dynamically assigned layers
 *
 * layer_count: Number of layers 
 * capacity: Number of layers the network can currently hold
 * input_features: Number of input features from the training data
 * output_classes: Number of classes or labels in the data
 * max_neurons: the maximum neurons of a single layer in the network
 * learning_rate: Networks learning rate
 * epochs: Number of epochs to train the network on
 * layers: dynamically allocated double array of network layers
 */
typedef struct {
    size_t layer_count;
    size_t capacity;

    size_t input_features;
    size_t output_classes;

    size_t output_layer_idx;
    size_t max_neurons;

    double learning_rate;
    size_t epochs;
    size_t seed;
    size_t threads;

    loss_t loss;
    layer_t *layers;
    
} network_t;


/* Evaluation type */

/*
 * Stores the results of evaluating a neural network on a labelled dataset.
 *
 * loss       - Loss calculated over the evaluation dataset.
 * threshold  - Binary classification decision threshold.
 *              Set to -1.0 for multiclass classification.
 * accuracy   - Proportion of correctly classified samples.
 * f1_score   - F1 score; positive-class F1 for binary classification,
 *              macro-averaged F1 for multiclass classification.
 * recall     - Recall; positive-class recall for binary classification,
 *              macro-averaged recall for multiclass classification.
 * precision  - Precision; positive-class precision for binary classification,
 *              macro-averaged precision for multiclass classification.
 * confusion  - Confusion matrix, with actual classes as rows and
 *              predicted classes as columns. Owned by this structure.
 *
 * Use free_evaluate() to release the confusion matrix and reset the structure.
 */
typedef struct {
    double loss;
    double threshold;

    double accuracy;
    double f1_score;
    double recall;
    double precision;

    matrix_t confusion;
} evaluate_t;

#endif
