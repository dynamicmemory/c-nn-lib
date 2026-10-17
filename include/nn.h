#ifndef NN_H
#define NN_H

#include <stddef.h>

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

/*
 * Creates a rows by cols matrix populated with random values
 * 
 * Params: 
 *   rows - number of rows in the matrix 
 *   cols - number of cols in the matrix 
 * 
 * Returns: 
 *   A constructed matrix or an empty matrix if allocation failures or dims 
 *   are invalid
*/
matrix_t random_matrix(size_t rows, size_t cols);

/*
 * Creates a rows by cols matrix initialized with zeros 
 * 
 * Params: 
 *   rows - number of rows in the matrix 
 *   cols - number of cols in the matrix 
 * 
 * Returns: 
 *   A constructed matrix or an empty matrix if allocation failures or dims 
 *   are invalid
*/
matrix_t zeros_matrix(size_t rows, size_t cols);

/*
 * Frees a matrix_t types allocated memory in its .data variable, resets the 
 * matrix to 0/NULL 
 * 
 * Params: 
 *   m - the matrix to free and cleanup the state of. 
*/
void free_matrix(matrix_t *m);


/* Function pointer types */
typedef void (*activation_fn)(const double *z, double *a, size_t n);
// typedef double (*dActivation_fn)(double a);
typedef void (*dActivation_fn)(const double *a, const double *dA, double *dZ, size_t n);

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



/* Loss function */
typedef double (*loss_fn)(layer_t layer, matrix_t y);
typedef void (*loss_gradient_fn)(layer_t layer, matrix_t y, matrix_t *dA);

typedef struct {
    loss_fn value;
    loss_gradient_fn gradient;
} loss_t;

extern const loss_t BINARY_CROSS_ENTROPY;
extern const loss_t CATEGORICAL_CROSS_ENTROPY;
/* 
 * Represents a neural network object, including its layers, training details 
 * loss function, and dynamically assigned layers
 *
 * layer_count: Number of layers 
 * capacity: Number of layers the network can currently hold
 * input_features: Number of input features from the training data
 * output_classes: Number of classes or labels in the data
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

    double learning_rate;
    size_t epochs;
    size_t seed;

    loss_t loss;
    layer_t *layers;

} network_t;

/* Prediction result — currently unused */
// typedef struct {size_t class; double *probabilities;
// } predict_t;


/* 
 * Creates a new neural network object
 *
 * Params: 
 *   features: Number of input features per samples
 *   lr:       Learning rate used during training
 *   epochs:   Maximum nmber of training epochs
 *   loss:     Loss function used to evaluate predicitons
*/
network_t neural_net(size_t features, double lr,
        size_t epochs, loss_t loss);

/*
 * Releases resources owned by the network, including layer parameters, 
 * gradients, and layer array. Does not free training data or other externally 
 * owned matrices, only network owned structures.
 *
 * Params:
 *   nn: Network to clean up, NULL is ignored.
 */
void cleanup(network_t *nn);

/*
 * Sets the networks random seed and reseeds the C standard libs process-wide 
 * pseudo-random number generator.
 *
 * Params:
 *   nn: Netowrk whose seed is being set 
 *   seed: seed value for the rng
 */
void set_seed(network_t *nn, size_t seed);


/*  
 * Adds a layer to a neural network. The layers input size is determined by the 
 * networks input features or the number of neurons from the prior layer. 
 * 
 * Parameters: 
 *   nn: network object to add the layer to 
 *   neurons: number of neurons for the layer 
 *   afunc: activation function for the layer
 * 
 * Returns: 
 *   0 on success and -1 on failure 
 */
int add_layer(network_t *nn, size_t neurons, activation_fn afunc);

/*   
 * Initialises a layer and allocates its parameter and gradient matrices. 
 * 
 * Parameters: 
 *   layer: layer structure to initialize 
 *   inputs: Number of input features per sample
 *   neurons: number of neurons for the layer 
 *   afunc: activation function for the layer
 * 
 * Returns: 
 *   0 on success and -1 on failure 
 */
int layer_init(layer_t *layer, size_t inputs, size_t neurons, activation_fn afunc);
dActivation_fn get_dActivation(activation_fn afunc);



/* 
 * Applies the ReLu activation element-wise 
 *
 * Params:
 *   z: Input values before activation
 *   a: output array for the activation
 *   n: number of elements
*/
void relu(const double *z, double *a, size_t n);

/* 
 * Applies the sigmoid activation element-wise 
 *
 * Params:
 *   z: Input values before activation
 *   a: output array for the activation
 *   n: number of elements
*/
void sigmoid(const double *z, double *a, size_t n);

/* 
 * Applies softmax across the input vector to produce a probability distro 
 *
 * Params:
 *   z: Input values before activation
 *   a: output array for probabilities 
 *   n: number of elements
*/
void softmax(const double *z, double *a, size_t n);

/*
 * Computes the ReLu derivative using the activation value
 *
 * Params:
 *   a: activation value
 *
 * Returns:
 *  1.0 if a > 0.0, otherwise 0.0
*/
// double d_relu(double a);
void d_relu(const double *a, const double *dA, double *dZ, size_t n);
/*
 * Computes the sigmoid derivative using the activation value
 *
 * Params:
 *   a: activation value
 *
 * Returns:
 *   the activated value of the input 'a' ~ a * (1.0 - a)
*/
// double d_sigmoid(double a);
void d_sigmoid(const double *a, const double *dA, double *dZ, size_t n);

/*
 * TODO: Determine the suitable interface, Not implemented currently 
 * NULL
*/
// void d_softmax(void);
void d_softmax(const double *a, const double *dA, double *dZ, size_t n);



/* 
 * TODO: Still resolving function 
 */
double binary_cross_entropy(layer_t layer, matrix_t y);

/* 
 * TODO: Still resolving function 
 */
double categorical_cross_entropy(layer_t layer, matrix_t y);

/* 
 * TODO: Still resolving function 
 */
void binary_cross_entropy_gradient(layer_t layer, matrix_t y, matrix_t *dA);

/* 
 * TODO: Still resolving function 
 */
void categorical_cross_entropy_gradient(layer_t layer, matrix_t y, matrix_t *dA);

/* 
 * Performs the forward pass for a single layer for the neural net. Calculates 
 * the weighted sum of the inputs plus bias for each neuron, then applies the  
 * activation function. 
 *
 * Params:
 *   layer: Ponter to the layer being evaluated
 *   x:     Input matrix with one sample per row and one feature per column
 *
 * Returns: 
 *   0 on success and -1 if memory failed to allocate for the Z scores
 */
int forward(layer_t *layer, matrix_t x);


int backprop(network_t *nn, matrix_t x, matrix_t y);
void update_weights_bias(network_t *nn);
void allocate_activations(layer_t *layer, matrix_t inputs);
void free_activations(network_t *nn);

/* Training, evaluation and prediction */
void train(network_t *nn, matrix_t x, matrix_t y);
void evaluate(network_t *nn, matrix_t x, matrix_t y);
void binary_evaluator(network_t *nn, matrix_t x, matrix_t y);
void predict(network_t *nn, matrix_t m, double *preds);
#endif
