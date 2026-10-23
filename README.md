# nn_lib (working title)

A neural network library written from scratch in C.

nn_lib implements a configurable feedforward neural network with backpropagation, 
gradient-based training, evaluation metrics, and multithreaded training. 
It is designed to make the underlying mechanics explicit: network layers, 
weight matrices, activation functions, loss functions, gradients, and parameter 
updates are implemented in C rather than delegated to a machine-learning framework.

The project includes an MNIST example that trains and evaluates a neural network 
on handwritten digit images.

## Features

- Configurable network architecture — construct networks dynamically by adding 
layers with different neuron counts and activation functions.

- Implemented from scratch in C — explicit forward propagation, backpropagation, 
gradient calculation, and parameter updates.

- Activation functions — ReLU, sigmoid, and softmax.

- Loss functions — binary cross-entropy and categorical cross-entropy.

- Training — full-batch gradient descent with configurable learning rate, 
epoch count, and optional loss-change-based early stopping.

- Parallel training — a data-parallel implementation using POSIX threads.

- Evaluation — loss, accuracy, precision, recall, F1 score, and confusion matrices.

- Prediction — obtain model outputs for new samples.

- Memory management — explicit allocation and cleanup through C functions.

- Testing — separate CTest targets for matrices, activations, losses, networks,
training, and evaluation.

- MNIST example — load handwritten digit data, train a classifier, 
inspect predictions, and evaluate performance.

## Project structure

```bash
nn_lib/
├── include/
│   ├── nnlib.h
│   ├── nnlib_types.h
│   ├── matrix.h
│   ├── activation.h
│   ├── loss.h
│   ├── network.h
│   ├── training.h
│   ├── evaluation.h
│   └── nnlib_internal.h
├── src/
│   ├── activation.c
│   ├── evaluation.c
│   ├── loss.c
│   ├── matrix.c
│   ├── network.c
│   ├── training.c
│   └── parallel_train.c
├── examples/
│   └── mnist/
│       ├── main.c
│       ├── mnist.c
│       ├── mnist.h
│       └── data/
├── test/
├── CMakeLists.txt
├── rebuild.sh
├── run.sh
└── test.sh
```

nnlib.h is the umbrella header for the public API. nnlib_types.h defines the 
data structures shared by the API. 

## Requirements

- A C compiler supporting C11.

- CMake 3.16 or later.

- POSIX threads.

- The standard math library.

The project uses CMake and links against libm and the platform's thread library.

## Build

From the project root:

```bash
cmake -S . -B build
cmake --build build
```
or use the provided script: rebuild.sh

To build with testing enabled:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
```
Run the registered tests with:

```bash
ctest --test-dir build --output-on-failure
```

or use the provided script: test.sh

The build creates the static library nn_lib and the mnist example executable.

## Quick start

The following example constructs a small network for binary classification.
It illustrates the public API; the input and target matrices must contain valid,
compatible data.

```c
#include "nnlib.h"

#include <stdio.h>

int main(void) {
    /* Two input features, learning rate 0.01, 1000 epochs. */
    network_t nn = neural_net(
        2,
        0.01,
        1000,
        BINARY_CROSS_ENTROPY
    );

    set_seed(&nn, 42);

    /* Hidden layer followed by a single binary output. */
    if (add_layer(&nn, 8, relu) < 0 ||
        add_layer(&nn, 1, sigmoid) < 0) {
        network_cleanup(&nn);
        return 1;
    }

    /*
     * x must have shape (samples, 2).
     * y must have shape (samples, 1), with binary targets.
     *
     * (For a multiclass problem, layers must be one hot encoded)
     * Create and populate these matrices before training using the matrix_t type
     */
    matrix_t x = {0};
    matrix_t y = {0};

    /*
     * Populate x and y with your training data here.
     */

    if (train(&nn, x, y, 1e-6, 1) < 0) {
        fprintf(stderr, "Training failed\n");
        free_matrix(&x);
        free_matrix(&y);
        network_cleanup(&nn);
        return 1;
    }

    evaluate_t results = {0};

    if (evaluate(&nn, x, y, &results, 0.5) == 0) {
        print_metrics(&results);
        print_confusion_matrix(results.confusion);
        free_evaluate(&results);
    }

    free_matrix(&x);
    free_matrix(&y);
    network_cleanup(&nn);

    return 0;
}
```

This example demonstrates the intended workflow:

1. Create a network with neural_net().

2. Optionally configure the random seed.

3. Add layers using add_layer().

4. Prepare input and target matrices.

5. Train using train().

6. Evaluate using evaluate().

7. Release evaluation results, data matrices, and network resources.

See docs/API.md for the complete API reference and data-format requirements.

## Network architecture

Networks are assembled dynamically rather than using fixed input, hidden,
or output layer counts.

For example, the MNIST handwritten digit classifier can use the following architecture:

```bash
Input: 784 features
        │
        ▼
Dense layer: 32 neurons
Activation: ReLU
        │
        ▼
Dense layer: 10 neurons
Activation: Softmax
        │
        ▼
Output: 10 class probabilities
```

Each layer stores its weights, biases, activations, and gradients. Weights are
represented by a matrix with shape (neurons, inputs), and biases by a matrix
with shape (neurons, 1).

The implementation uses row-major matrices and computes each sample's neuron 
activations from the corresponding input row.

## Training

Training uses full-batch gradient descent:

1. Run a forward pass through each layer.

2. Calculate the loss from the output layer.

3. Backpropagate gradients through the network.

4. Average the weight and bias gradients over the training samples.

5. Update the weights and biases using the configured learning rate.

6. Optionally stop when the absolute change between successive epoch losses falls
below the specified tolerance.

A tolerance of 0.0 disables tolerance-based early stopping under the current 
implementation, so training runs for the configured number of epochs.

The threads field in network_t controls the selected training path:

```bash
nn.threads = 1;  /* Serial training */
nn.threads = 4;  /* Parallel training path */
```

Serial training is the default. Parallel training partitions the training 
samples across workers, computes local gradients, reduces them into full-batch
gradients, and updates the shared model parameters.

Parallel training can significantly speed up your network training by a factor 5, 
you can read more about it at docs/nnlib-benchmark-report.md.

## Loss functions and activations

The library currently provides:

| Task | Output activation | Loss |
| :--- | :--- | :--- |
| Binary classification | sigmoid | BINARY_CROSS_ENTROPY |
| Multiclass classification | softmax | CATEGORICAL_CROSS_ENTROPY |

relu and sigmoid are available for hidden layers. The library exposes the loss and activation 
functions through function pointers and loss_t structures.

These are the intended pairings; the current API does not automatically validate
that a network's output activation matches its loss function. 

## Evaluation

evaluate() computes the configured loss and classification metrics using 
predictions from the network.

For binary classification, predictions are classified using a configurable
threshold, default is set to 0.5. The reported precision, recall, and F1 score
refer to the positive class.

For multiclass classification, predictions use the class with the highest 
output value. Precision, recall, and F1 are macro-averaged across classes.

The confusion matrix uses actual classes as rows and predicted classes as columns.

predict() returns the network's output values in a newly allocated matrix.
For a softmax output layer, these values represent class probabilities. Check 
the API.md file to understand the return types and how to interpret them.

## MNIST example

The included example demonstrates multiclass classification using the MNIST 
handwritten digit dataset. It loads image and label files, normalises image 
pixels into the range [0, 1], one-hot encodes labels, trains a network, evaluates 
the training and test sets, and prints an example prediction.

The example expects the four standard MNIST IDX files in examples/mnist/data/:

```bash
train-images-idx3-ubyte
train-labels-idx1-ubyte
t10k-images-idx3-ubyte
t10k-labels-idx1-ubyte
```

Run the example from the project root so its relative data paths resolve correctly:

```bash
./run.sh
```

The example's sample count, epoch count, learning rate, and thread count are 
configured in examples/mnist/main.c.

## Memory ownership

The API uses explicit C memory management.

- neural_net() returns a network value that owns dynamically allocated layer 
storage as layers are added.

- network_cleanup() releases the network's layer matrices and layer array.

- zeros_matrix() and random_matrix() return matrices whose allocated data must 
be released using free_matrix().

- predict() allocates the output matrix; the caller releases it using free_matrix().

- evaluate() allocates the confusion matrix in its result; the caller releases 
it using free_evaluate().

- Input and target matrices supplied to training, prediction, and evaluation
remain caller-owned.

Do not shallow-copy an owning network_t or independently free its internal
matrix pointers. Initialise output structures appropriately, and do not reuse
an evaluation result that still owns a confusion matrix without freeing it first.

## Current scope and limitations

This is a learning-oriented implementation of a feedforward neural network,
not a complete machine-learning framework.

Current limitations include:

- Full-batch training only; no minibatch training or shuffling.

- No automatic validation of all matrix dimensions, target values, or 
activation/loss combinations.

- No GPU support.

- No convolutional or recurrent layers.

- No model serialization or checkpointing API. (coming soon)

- The random seed uses the C standard library's process-wide random-number generator.

- Parallel training still needs more robust handling of worker creation failures,
worker errors, and resource cleanup.

- Some allocation and invalid-input failure paths require additional hardening.

The implementation was intended as a learning exercise and grew into a transparent 
foundation for experimenting with neural network internals and extending the 
library over time. If anyone ever gets this far... thank you, this was alot of 
fun to build.

## License

This project is licensed under the MIT Liscense. See the [LICENSE](docs/LICENSE) file for details.
