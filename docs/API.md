# nnlib API Reference

This document describes the current stable public API exposed by include/nnlib.h and the 
associated public headers. This document was written with the help of qwen3.8:27b 
and is to be treated as a first version with possible revisions and refinements 
to be added in the future.

The libraray is written in C. Public structures are define in nnlib_types.h and 
the functions are grouped by responsibility across the matrix, activation, loss, 
network, training, and evaluation interfaces.

## Table of contents 

* [Conventions and ownership](#conventions-and-ownership)
* [Data types](#data-types)
* [Matrix functions](#matrix-functions)
* [Activation functions](#activation-functions)
* [Loss functions](#loss-functions)
* [Network functions](#network-functions)
* [Training](#training)
* [Prediction and evaluation](#prediction-and-evaluation)
* [Recommended configurations](#recommended-configurations)
* [Current limitations](#current-limitations)

## Conventions and ownership 

Include the public header: 

```C 
#include "nnlib.h"
```

Unless otherwise stated:

- Matrices use row-major storage
- Functions returning int use 0 for success and -1 for failure 
- Matrix arguments are passed by value, but their data pointers refer to the 
underlying allocated storage 
- Functions that allocate output matrices transfer ownership of the allocated  
storage to the caller
- The caller is responsible for realeasing owned resources with the corresponding 
cleanup functions

### Matrix layout 

A ```bash matrix_t ``` stores rows * cols doubles contiguously. The element at 
row r and the column c is accessed using:

```c
matrix.data[r * matrix.cols + c]
```

The size field records the number of elements.

### Ownership rules

| Resource | Owner | Cleanup | 
| :--- | :--- | :--- |
| Matrix returned by zeros_matrix() or random_matrix() | Caller | free_matrix() |
| Input and target matrices passed to the network | Caller | free_matrix() if dynamically allocated | 
| Network layer storage | Network | network_cleanup() |
| Prediction matrix returned by predict() | Caller | free_matrix() | 
| Confusion matrix in evaluate_t | Evaluation result owner | free_evaluate() |

Do not shallow-copy an owning network_t and then clean up both copies. The 
structure contains pointers to heap-allocated memory.

## Data types

### matrix_t

```c
typedef struct {
    size_t rows;
    size_t cols;
    size_t size;
    double *data;
} matrix_t;
```

Represents a row-major matrix of doubles.

- rows: number of rows.
- cols: number of columns.
- size: number of elements.
- data: pointer to contiguous matrix storage.

An empty or failed allocation is represented by a zero-initialised matrix, typically {0}.

### activation_fn

```c
typedef void (*activation_fn)(
    const double *z,
    double *a,
    size_t n
);
```

A function pointer for an activation function. The function reads n input values and writes n activated values.

### dActivation_fn

```c
typedef void (*dActivation_fn)(
    const double *a,
    const double *dA,
    double *dZ,
    size_t n
);
```

A function pointer for propagating a gradient through an activation function.

### layer_t

```c
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
```

Represents a fully connected layer.

| Field | Meaning |
| :--- | :--- |
| inputs | Number of inputs per sample |
| neurons | Number of neurons in the layer |
| W | Weight matrix, shape (neurons, inputs) |
| bias | Bias matrix, shape (neurons, 1) |
| a | Activations for the current batch, shape (samples, neurons) |
| dW | Weight gradients, shape (neurons, inputs) |
| dB | Bias gradients, shape (neurons, 1) |
| activation | Activation function | 
| d_activation | Activation-gradient function |

These fields are public, but the library expects their dimensions and allocated 
storage to remain consistent with the layer's configuration.

### loss_t

```c
typedef double (*loss_fn)(layer_t layer, matrix_t y);

typedef void (*loss_gradient_fn)(
    layer_t layer,
    matrix_t y,
    matrix_t *dA
);

typedef struct {
    loss_fn value;
    loss_gradient_fn gradient;
} loss_t;
```

Combines a loss calculation and its corresponding gradient calculation.

The library supplies BINARY_CROSS_ENTROPY and CATEGORICAL_CROSS_ENTROPY. Custom 
loss functions can be represented using the same structure, but must obey the 
expected dimensions and gradient conventions of the implementation.

### network_t

```c
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
```

Stores the network architecture, configuration, and owned layer array.

| Field | Meaning |
| :--- | :--- |
| layer_count | Number of configured layers |
| capacity | Allocated capacity of the layer array |
| input_features | Number of input features per sample |
| output_classes | Neuron count in the most recently added layer |
| output_layer_idx | Index of the output layer |
| max_neurons | Maximum neuron count across configured layers |
| learning_rate | Gradient-descent learning rate |
| epochs | Maximum training epochs
| seed | Configured random seed |
|threads | Training thread count; defaults to 1 |
|loss | Loss calculation and gradient functions |
|layers | Dynamically allocated layer array |

output_classes is the number of neurons in the final layer, including for a single-neuron binary output. It does not mean that a binary classifier has that many distinct output classes.

### evaluate_t

```c
typedef struct {
    double loss;
    double threshold;
    double accuracy;
    double f1_score;
    double recall;
    double precision;
    matrix_t confusion;
} evaluate_t;
```

Contains the evaluation results.

- loss: loss computed from the predicted outputs.
- threshold: binary decision threshold, or -1.0 for multiclass evaluation.
- accuracy: proportion of correct predictions.
- precision: binary positive-class precision or multiclass macro precision.
- recall: binary positive-class recall or multiclass macro recall.
- f1_score: binary F1 or multiclass macro-averaged per-class F1.
- confusion: confusion matrix, with actual classes as rows and predicted classes as columns.

The confusion matrix is owned by the result and must be released using free_evaluate().

## Matrix functions

Declared in matrix.h.

### zeros_matrix()

```c
matrix_t zeros_matrix(size_t rows, size_t cols);
```

Allocates a matrix initialised to zero.

Parameters

- rows: number of rows; must be nonzero.
- cols: number of columns; must be nonzero.

Returns

A matrix with allocated zero-initialised storage, or an empty matrix if 
dimensions are invalid, the allocation size overflows, or allocation fails.

Ownership: the caller must release a successfully allocated matrix using free_matrix().

### random_matrix()

```c
matrix_t random_matrix(size_t rows, size_t cols);
```

Allocates a matrix populated with pseudo-random values generated using rand(). 
Values are drawn using the current C standard library random state and scaled 
approximately to [-0.1, 0.1]. The dimension and allocation-failure behaviour 
matches zeros_matrix().

### free_matrix()

```c
void free_matrix(matrix_t *m);
```

Frees a matrix's data pointer and resets the structure to an empty matrix.

Passing NULL is safe.

Use this function for matrices whose allocated storage is owned by the caller. 
Do not use it on a non-owning view into another matrix's storage.

## Activation functions

Declared in activation.h.

All activation functions operate on arrays of length n and write their outputs to a caller-provided array.

### relu()

```c
void relu(const double *z, double *a, size_t n);
```

Applies the rectified linear unit elementwise:

### sigmoid()

```c
void sigmoid(const double *z, double *a, size_t n);
```

Applies the logistic sigmoid:

The implementation uses separate branches for positive and negative inputs to improve numerical stability.

### softmax()

```c
void softmax(const double *z, double *a, size_t n);
```

Converts a vector of logits into normalised exponential outputs:

Subtracting the maximum input reduces overflow risk. The function assumes a nonempty vector and valid pointers.

### d_relu()

```c
void d_relu(
    const double *a,
    const double *dA,
    double *dZ,
    size_t n
);
```

Propagates gradients through ReLU. The implementation returns zero where the 
activation is not positive and passes through the incoming gradient where it is positive.

### d_sigmoid()

```c
void d_sigmoid(
    const double *a,
    const double *dA,
    double *dZ,
    size_t n
);
```

Propagates gradients through sigmoid:


### d_softmax()

```c
void d_softmax(
    const double *a,
    const double *dA,
    double *dZ,
    size_t n
);
```

Propagates gradients through softmax using the Jacobian-vector product:

The implementation computes this product without explicitly constructing the full Jacobian matrix.

### get_dActivation()

```c
dActivation_fn get_dActivation(activation_fn afunc);
```

Returns the corresponding derivative function for the recognised activation 
function pointer (relu, sigmoid, or softmax). Returns NULL for an unsupported function pointer.

The function identifies activations by comparing function pointers. Arbitrary 
user-defined activations are not automatically supported for backpropagation.

## Loss functions

Declared in loss.h.

### BINARY_CROSS_ENTROPY

```c
extern const loss_t BINARY_CROSS_ENTROPY;
```

A predefined loss object containing the binary cross-entropy calculation and its gradient.

Use with a single-output network, conventionally using sigmoid activation. 
Targets should have shape (samples, 1) and contain binary values.

The loss clips output probabilities to avoid taking the logarithm of zero.

### CATEGORICAL_CROSS_ENTROPY

```c
extern const loss_t CATEGORICAL_CROSS_ENTROPY;
```

A predefined categorical cross-entropy loss object.

Use with a multiclass output layer, conventionally using softmax activation. 
Targets should have shape (samples, classes) and normally be one-hot encoded.

### binary_cross_entropy()

```c
double binary_cross_entropy(layer_t layer, matrix_t y);
```

Calculates mean binary cross-entropy using layer.a as the predicted probabilities and y as the target matrix.

The output layer and target matrix must have compatible dimensions. The function assumes at least one sample.

### categorical_cross_entropy()

```c
double categorical_cross_entropy(layer_t layer, matrix_t y);
```

Calculates mean categorical cross-entropy from predicted probabilities and target values.

### binary_cross_entropy_gradient()

```c
void binary_cross_entropy_gradient(
    layer_t layer,
    matrix_t y,
    matrix_t *dA
);
```

Writes the derivative of the binary cross-entropy loss with respect to the predicted activation values into dA.

The gradient function does not average over the batch itself. The 
backpropagation implementation averages weight and bias gradients over the training batch.

### categorical_cross_entropy_gradient()

```c
void categorical_cross_entropy_gradient(
    layer_t layer,
    matrix_t y,
    matrix_t *dA
);
```

Writes the categorical cross-entropy activation gradient into dA. It expects 
compatible target and output dimensions.

The supplied loss functions do not perform comprehensive dimension or target 
validation. Pass valid data and use the intended activation/loss pairing.

## Network functions

Declared in network.h.

### neural_net()

```c
network_t neural_net(
    size_t features,
    double lr,
    size_t epochs,
    loss_t loss
);
```

Initialises a network configuration.

Parameters

- features: number of input features per sample.
- lr: learning rate.
- epochs: maximum number of training epochs.
- loss: loss object used for training and evaluation.

The constructor initialises an empty network with no layers and sets threads to 1.

Returns: a network_t value. Layers and their matrices are allocated as layers are added.

The constructor seeds the process-wide C random-number generator. Validate the 
configuration and add layers before training.

### set_seed()

```c
void set_seed(network_t *nn, size_t seed);
```

Sets the network's recorded seed and calls srand() with that seed.

This affects the process-wide C random-number generator, not a private generator 
owned by the network. Calls to other code using rand() can therefore affect reproducibility.

Passing a null network pointer produces an error message and returns without changing anything.

### add_layer()

```c
int add_layer(
    network_t *nn,
    size_t neurons,
    activation_fn afunc
);
```

Adds a fully connected layer to the network.

Parameters

- nn: network to modify.
- neurons: number of neurons in the new layer.
- afunc: activation function, currently one of relu, sigmoid, or softmax.

The first layer's input count is taken from nn->input_features. Subsequent layers 
take their input count from the previous layer's neuron count. Weights, biases, 
and gradient matrices are allocated for the new layer.

Returns: 0 on success, -1 on failure.

A zero-neuron layer, invalid network configuration, or incompatible loss/activation pairing is not comprehensively validated. Use positive dimensions and a valid configuration.

### network_cleanup()

```c
void network_cleanup(network_t *nn);
```

Releases the layer matrices and layer array owned by the network.

Call this once the network is no longer needed. It does not free external input or target matrices.

The current implementation resets the layer array and its count/capacity, but 
does not reset every configuration field. Treat cleanup as releasing network-owned 
resources, rather than as a complete reinitialisation of a network object.

Training

Declared in training.h.

### train()

```c
int train(
    network_t *nn,
    matrix_t x,
    matrix_t y,
    double tol,
    int verbose
);
```

Trains the network using full-batch gradient descent.

Parameters

- nn: configured network with at least one layer.
- x: input matrix with shape (samples, input_features).
- y: target matrix with the same number of rows as x.
- tol: absolute loss-change tolerance used for early stopping.
- verbose: nonzero to print epoch loss information.

Target shapes:

| Task | Output neurons | Target shape |
| :--- | :--- | :--- |
| Binary classification | 1 | (samples, 1) |
| Multiclass classification | Number of classes | (samples, classes) |

Binary targets should contain 0 or 1. Multiclass targets should normally be one-hot encoded.

#### Training behaviour

Each epoch computes the loss before updating the parameters, calculates gradients 
by backpropagation, and updates weights and biases.

If tol > 0, training stops when the absolute difference between successive epoch 
losses is less than tol. A tolerance of 0 disables early stopping in the current implementation.

#### Thread selection

nn->threads <= 1: serial training.

nn->threads > 1: parallel training.

Returns: 0 on successful completion, -1 when the selected training path reports an error.

The function currently performs limited input validation. The caller must supply 
compatible, nonempty matrices and a correctly configured network.

#### Parallel-training caveats

The parallel path splits samples between workers, computes local gradients, averages 
those gradients by each worker's share of the training data, and updates the main network.

## Prediction and evaluation

Declared in evaluation.h.

### predict()

```c
int predict(
    network_t *nn,
    matrix_t x,
    matrix_t *preds
);
```

Runs a forward pass and allocates an output matrix containing the final layer's activations.

Parameters

- nn: configured network.
- x: input matrix with shape (samples, input_features).
- preds: output pointer receiving the newly allocated prediction matrix.

The output shape is (samples, output_neurons). For a softmax output layer, each 
row contains the class probabilities.

Returns: 0 on success, -1 on reported failure.

On success, the caller owns the allocated matrix and must release it using 
free_matrix(preds). Initialise the output variable to {0} before use. The 
current implementation does not validate all dimensions or clean up every 
intermediate allocation on every failure path.

### evaluate()

```c
int evaluate(
    network_t *nn,
    matrix_t x,
    matrix_t y,
    evaluate_t *result,
    double threshold
);
```

Calculates the loss, confusion matrix, accuracy, precision, recall, and F1 score.

Parameters

- nn: configured network.
- x: input matrix.
- y: target matrix.
- result: pointer to an evaluation structure that will receive the results.
- threshold: binary classification threshold; conventionally 0.5.

For binary classification, the output must contain one neuron and targets must 
have one column. Predictions at or above the threshold are classified as positive.

For multiclass classification, targets must have one column per output neuron. 
The predicted class is selected using argmax; the supplied threshold is ignored and result->threshold is set to -1.0.

Metrics

- Accuracy: fraction of samples correctly classified.
- Precision: positive-class precision for binary classification; macro precision for multiclass classification.
- Recall: positive-class recall for binary classification; macro recall for multiclass classification.
- F1: positive-class F1 for binary classification; macro-averaged classwise F1 for multiclass classification.
- Loss: the configured loss applied to predicted activations and target values.

The confusion matrix has actual classes on rows and predicted classes on columns. 
Binary class index 0 is negative and class index 1 is positive.

Returns: 0 on success, -1 on reported failure.

Ownership: the result owns the allocated confusion matrix after a successful 
call. Initialise a fresh result as evaluate_t result = {0}; and call 
free_evaluate(&result) before discarding it. Do not overwrite a result that still owns a confusion matrix, because the current implementation does not free a previous result automatically.

The function checks some input conditions, but it does not comprehensively validate threshold ranges or target values.

### print_confusion_matrix()

```c
void print_confusion_matrix(matrix_t confusion);
```

Prints the confusion matrix to standard output, with actual classes represented by rows and predicted classes by columns. The current output uses ANSI escape sequences for cell colouring.

### print_metrics()

```c
void print_metrics(const evaluate_t *eval);
```

Prints the loss and classification metrics. Binary evaluation includes the decision threshold; multiclass evaluation indicates argmax classification.

### free_evaluate()

```c
void free_evaluate(evaluate_t *eval);
```

Releases the confusion matrix and resets the evaluation structure.

Passing a null pointer is safe.

Recommended configurations

### Binary classification

```c
network_t nn = neural_net(
    input_features,
    0.01,
    1000,
    BINARY_CROSS_ENTROPY
);
```

add_layer(&nn, 16, relu);
add_layer(&nn, 1, sigmoid);

Use target values with shape (samples, 1).

### Multiclass classification

```c
network_t nn = neural_net(
    input_features,
    0.01,
    1000,
    CATEGORICAL_CROSS_ENTROPY
);

add_layer(&nn, 32, relu);
add_layer(&nn, number_of_classes, softmax);
```

Use one-hot target matrices with shape (samples, number_of_classes).

These examples illustrate intended configurations. Check return values from 
layer creation and training, and call network_cleanup() when finished.

## Current limitations

The API and implementation currently have several limitations that callers should account for:

- Training uses full batches, with no minibatch or shuffle interface.
- Matrix dimensions and target values are not validated consistently across functions.
- Activation/loss compatibility is a caller responsibility.
- The C standard library random-number generator is process-wide.
- Allocation failure paths in training and evaluation are not all leak-free.
- The library does not currently provide serialization, checkpointing, GPU execution, or additional layer types.

Most of these issue will be resolved in time, thought there is no exact timeline of when 



