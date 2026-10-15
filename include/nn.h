#ifndef NN_H
#define NN_H

#include <stddef.h>

/* Function pointer types */
typedef void (*activation_fn)(const double *z, double *a, size_t n);
typedef double (*dActivation_fn)(double a);

/* Matrix */
typedef struct {
    size_t rows;
    size_t cols;
    size_t size;
    double *data;
} matrix_t;

/* Neural-network layer */
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

/* Loss function */
typedef double (*loss_fn)(layer_t layer, matrix_t y);

/* Neural network */
typedef struct {
    size_t layer_count;
    size_t capacity;


    size_t input_features;
    size_t output_classes;

    size_t input_layer;
    size_t output_layer_idx;

    double learning_rate;
    size_t epochs;
    size_t seed;

    loss_fn loss;
    layer_t *layers;

} network_t;

/* Prediction result — currently unused */
// typedef struct {size_t class; double *probabilities;
// } predict_t;

/* Network construction and memory management */
void neural_net(network_t *nn, size_t features, double lr,
        size_t epochs, loss_fn loss);
void add_layer(network_t *nn, size_t neurons, activation_fn afunc);
layer_t layer(size_t inputs, size_t neurons, activation_fn afunc);
void cleanup(network_t *nn, matrix_t x, matrix_t y);
void set_seed(network_t *nn, double seed);

/* Matrix creation */
matrix_t random_matrix(size_t inputs, size_t neurons);
matrix_t zeros_matrix(size_t inputs, size_t neurons);

/* Activations and derivatives */
void relu(const double *z, double *a, size_t n);
void sigmoid(const double *z, double *a, size_t n);
void softmax(const double *z, double *a, size_t n);
double d_relu(double a);
double d_sigmoid(double a);
void d_softmax(void);
dActivation_fn get_dActivation(activation_fn afunc);

/* Losses */
double binary_cross_entropy(layer_t layer, matrix_t y);
double categorical_cross_entropy(layer_t layer, matrix_t y);

/* Forward and backward propagation */
void forward(layer_t *layer, matrix_t x);
void backprop(network_t *nn, matrix_t x, matrix_t y);
void update_weights_bias(network_t *nn);
void allocate_activations(layer_t *layer, matrix_t inputs);
void free_activations(network_t *nn);

/* Training, evaluation and prediction */
void train(network_t *nn, matrix_t x, matrix_t y);
void evaluate(network_t *nn, matrix_t x, matrix_t y);
void binary_evaluator(network_t *nn, matrix_t x, matrix_t y);
void predict(network_t *nn, matrix_t m, double *preds);
#endif
