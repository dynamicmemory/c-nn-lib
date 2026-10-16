/* TODO: PREDICT: Make it work, for binary and multiclass, consider regression 
          for the future too.*/
// TODO: Onehotencoding for labels... decide if always must and how to handle user
// TODO: Enum for activation / deactivation version to not pass in decactive 
// TODO: weight decay regularization param for a
// TODO: Fix consts throughout the project
#include "nn.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>


// Loss functions
double binary_cross_entropy(layer_t layer, matrix_t y) {
    double loss = 0.0;
    double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row) {  
        double a = layer.a.data[row];
        if (a < epsilon) a = epsilon;
        if (a > 1.0 - epsilon) a = 1.0 - epsilon;
        loss -= y.data[row] * log(a) + (1.0 - y.data[row]) * log(1.0 - a); 
    }
    loss /= y.rows;
    return loss;
}

double categorical_cross_entropy(layer_t layer, matrix_t y) {
    double loss = 0.0;
    const double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row)
        for (size_t neuron=0; neuron<layer.neurons; ++neuron) {
            size_t idx = row * layer.neurons + neuron;

            if (y.data[idx] != 0.0) {
                double a = layer.a.data[idx];
                if (a < epsilon) a = epsilon;

                loss -= y.data[idx] * log(a);
            }
        }
    return loss /y.rows;
}


void forward(layer_t *layer, matrix_t x) {
    double *Z = calloc(layer->neurons, sizeof(double));

    for (size_t row=0; row<x.rows; ++row) {
        for (size_t neuron=0; neuron<layer->neurons; ++neuron) {
            size_t a_idx = row * layer->neurons + neuron;
            double z = layer->bias.data[neuron];

            for (size_t input=0; input<layer->inputs; ++input) {
                size_t w_idx = neuron * layer->W.cols + input;
                size_t x_idx = row * x.cols + input;

                z += layer->W.data[w_idx] * x.data[x_idx]; 
            }
            Z[neuron] = z;
        }
        layer->activation(Z, &layer->a.data[row * layer->neurons], layer->neurons); 
    }
    free(Z);
}

void backprop(network_t *nn, matrix_t x, matrix_t y) {
    layer_t *layer = nn->layers;

    // Reset gradients 
    for (size_t i=0; i<nn->layer_count; ++i) {
        memset(layer[i].dB.data, 0, layer[i].dB.rows * layer[i].dB.cols * sizeof(double));
        memset(layer[i].dW.data, 0, layer[i].dW.rows * layer[i].dW.cols * sizeof(double));
    }

    matrix_t dZ = {0}, dA = {0};
    for (size_t i=nn->layer_count; i-->0; ) {
        dZ.rows = x.rows;
        dZ.cols = nn->layers[i].neurons;
        dZ.data = malloc(dZ.rows * dZ.cols * sizeof(double));

        // Calculating dZ: if output layer, else hidden/input
        if (i == nn->output_layer_idx)
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                    size_t dZ_idx = row * layer[i].neurons + neuron;
                    // TODO: Assumption - bin/categorical cross entropy + sigmoid/softmax
                    dZ.data[dZ_idx] = layer[i].a.data[dZ_idx] - y.data[dZ_idx];
                }
        else {
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                    size_t dZ_idx = row * layer[i].neurons + neuron;
                    size_t dA_idx = row * dA.cols + neuron;
                    dZ.data[dZ_idx] = dA.data[dA_idx] * layer[i].d_activation(layer[i].a.data[dZ_idx]); 
                }
            free(dA.data);
        }

        // Calculate dW: Input layer, else hidden/output
        matrix_t *input_ptr = (i==0) ? &x : &layer[i-1].a;
        for (size_t row=0; row<input_ptr->rows; ++row)
            for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                layer[i].dB.data[neuron] += dZ.data[row * layer[i].neurons + neuron];

                for (size_t input=0; input<layer[i].inputs; ++input) {
                    size_t w_idx = neuron * layer[i].W.cols + input;
                    size_t a_idx = row * input_ptr->cols + input;
                    size_t dZ_idx = row * layer[i].neurons + neuron;

                    layer[i].dW.data[w_idx] += input_ptr->data[a_idx] * dZ.data[dZ_idx];
                }
            }

        // Calculate dA for current row, for use in the next layers dZ calculations
        if (i > 0) {
            dA.rows=x.rows;
            dA.cols=layer[i].W.cols;
            dA.data= calloc(dA.rows * dA.cols, sizeof(double));
            
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron)
                    for (size_t input=0; input<layer[i].inputs; ++input) {
                        size_t w_idx = neuron * layer[i].W.cols + input;
                        size_t dA_idx = row * dA.cols + input;
                        size_t dZ_idx = row * layer[i].neurons + neuron;

                        dA.data[dA_idx] += dZ.data[dZ_idx] * layer[i].W.data[w_idx];
                    }
        }
        free(dZ.data);
    }

    // TODO: Average the gradient losses to match our current loss function evaluation
    for (size_t i=0; i<nn->layer_count; ++i) {
        for (size_t j = 0; j< layer[i].dW.cols * layer[i].dW.rows; ++j)
            layer[i].dW.data[j] /= x.rows;
        for (size_t j = 0; j< layer[i].dB.cols * layer[i].dB.rows; ++j)
            layer[i].dB.data[j] /= x.rows;
    }
}

void update_weights_bias(network_t *nn) {
    layer_t *layer = nn->layers;
    double lr = nn->learning_rate;

    for (size_t i=0; i<nn->layer_count; ++i)
        for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
            layer[i].bias.data[neuron] -= lr * layer[i].dB.data[neuron];
            for (size_t input=0; input<layer[i].inputs; ++input) {
                size_t w_idx = neuron * layer[i].W.cols + input;
                layer[i].W.data[w_idx] -= lr * layer[i].dW.data[w_idx];
            }
        }
}

void train(network_t *nn, matrix_t x, matrix_t y){
    double prior_loss = INFINITY;
    double loss, delta, change;
    double epsilon = 1e-6;

    layer_t *layer = nn->layers;
    int count = 0; // TODO: Count be a network variable for a user to call
    while (nn->epochs-- > 0) {
        matrix_t *input = &x;
        for (size_t i=0; i<nn->layer_count; ++i) {
            allocate_activations(&layer[i], *input);
            forward(&layer[i], *input);
            input = &layer[i].a;
        }

        loss = nn->loss(layer[nn->output_layer_idx], y);
        backprop(nn, x, y);
        update_weights_bias(nn);

        delta = loss-prior_loss;
        change = delta < 0 ? delta * -1 : delta;
        if (change < epsilon)
            break;
        prior_loss = loss;

        // TODO: This needs to be user controlled
        // if (count % 500 == 0)
            printf("Epoch: %d | Loss: %f\n", count, loss);
        count++;
    }
    free_activations(nn);
}

void allocate_activations(layer_t *layer, matrix_t inputs) {
    if (layer->a.data != NULL)
        return;

    layer->a.rows = inputs.rows;
    layer->a.cols = layer->neurons;
    layer->a.data = malloc(layer->a.rows * layer->a.cols * sizeof(double));
}

void free_activations(network_t *nn) {
    layer_t *layer = nn->layers;

    for (size_t i=0; i<nn->layer_count; ++i) {
        free(layer[i].a.data);
        layer[i].a.data = NULL;
    }
}

// TODO: x y size match
void binary_evaluator(network_t *nn, matrix_t x, matrix_t y) {
    double *preds = malloc(x.rows * y.cols * sizeof(double));
    predict(nn, x, preds);
    size_t pred = 0;

    for (size_t row=0; row<y.rows; ++row) { 
        size_t best_y = 0;

        pred = preds[row] > 0.5 ? 1 : 0;
        best_y = y.data[row];

        printf("a[%zu]: %f ", row, preds[row]);
    printf("Target: %zu | Predicted: %zu\n", best_y, pred);
    }
    // TODO: As it stands we must free preds here, but this will change so a user 
    // can use the returned values or we will store conf mat + metrics somewhere prior to freeing here 
    free(preds);
}

void evaluate(network_t *nn, matrix_t x, matrix_t y) {
    double *preds = malloc(x.rows * y.cols * sizeof(double));
    predict(nn, x, preds);

    // TODO: Temp accuracy measure 
    double accuracy, pos, neg;
    accuracy = pos = neg = 0;

    for (int row=0; row<y.rows; ++row) { 
        size_t best = 0;
        size_t best_y = 0;

        for (size_t c=0; c<nn->output_classes; ++c) {

            size_t current_class_idx = row * nn->output_classes + c;
            size_t best_class_idx = row * nn->output_classes + best;
            size_t current_target_class = row * y.cols + c;
            size_t predicted_target_class = row * y.cols + best_y;

            if (preds[current_class_idx] > preds[best_class_idx])
                best = c;
            if (y.data[current_target_class] != 0) 
                best_y = c;

            // printf("a[%d]: %f ", row, preds[current_class_idx]);
        }
        // printf("Target: %zu | Predicted: %zu\n", best_y, best);
        if (best_y == best) 
            pos++;
        else 
            neg++;
    }
    printf("Accuracy: %f\n", pos/(pos+neg));
    // TODO: As it stands we must free preds here, but this will change so a user 
    //       can use the returned values or we will store conf mat + metrics somewhere prior to freeing here 
    free(preds);
}

void predict(network_t *nn, matrix_t m, double *preds) {
    layer_t *layer = nn->layers;
    matrix_t *input = &m;

    for (size_t i=0; i<nn->layer_count; ++i) {
        allocate_activations(&layer[i], *input);
        forward(&layer[i], *input);
        input = &layer[i].a;
    }
    for (size_t i=0; i<m.rows * nn->output_classes; ++i)
        preds[i] = layer[nn->output_layer_idx].a.data[i];

    free_activations(nn);
}


// matrix.c
matrix_t random_matrix(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols)
        return (matrix_t){0}; 

    size_t size = rows * cols;
    if (size > SIZE_MAX / sizeof(double))
        return (matrix_t){0}; 

    matrix_t w = {
        .rows = rows, 
        .cols = cols,
        .size = size,
        .data = malloc(size * sizeof(double))
    };

    if (w.data == NULL) 
        return (matrix_t){0}; 

    for (size_t i=0; i<w.size; ++i)
        w.data[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    return w;
}

matrix_t zeros_matrix(size_t rows, size_t cols) {
    if (rows == 0 || cols == 0 || rows > SIZE_MAX / cols)
        return (matrix_t){0}; 

    size_t size = rows * cols;
    if (size > SIZE_MAX / sizeof(double))
        return (matrix_t){0}; 

    matrix_t w = {
        .rows = rows, 
        .cols = cols,
        .size = size,
        .data = calloc(size, sizeof(double))
    };

    if (w.data == NULL) 
        return (matrix_t){0}; 

    return w;
}

void free_matrix(matrix_t *m) {
    if (m == NULL) 
        return;

    free(m->data);
    *m = (matrix_t){0};
}

int add_layer(network_t *nn, size_t neurons, activation_fn afunc) {
    if (nn == NULL) {
        fprintf(stderr, "add_layer: Network not initialize\n");
        return -1;
    }

    if (nn->layer_count == nn->capacity) {
        size_t new_cap = nn->capacity == 0 ? 2 : nn->capacity * 2;

        layer_t *new_layers = realloc(nn->layers, new_cap * sizeof(layer_t));
        if (new_layers == NULL) {
            fprintf(stderr, "add_layer: Layers failed to reallocate\n");
        return -1;
        }

        nn->layers = new_layers;
        nn->capacity = new_cap;
    }

    // Get the input size for the layer, from the prior layers output or input features
    size_t input_size = nn->input_features;

    if (nn->layer_count > 0)
        input_size = nn->layers[nn->layer_count - 1].neurons;

    // Add layer
    layer_t *new_layer = &nn->layers[nn->layer_count];
    if (layer_init(new_layer, input_size, neurons, afunc) < 0) { 
        fprintf(stderr, "add_layer: Failed to create new layer\n");
        return -1;
    }
    nn->layer_count++;
    nn->output_classes = neurons;
    nn->output_layer_idx = nn->layer_count - 1;
    return 0;
}

int layer_init(layer_t *layer, size_t inputs, size_t neurons, activation_fn afunc) {
    *layer = (layer_t){0};

    layer->inputs = inputs;
    layer->neurons = neurons; 
    layer->activation = afunc;
    layer->d_activation = get_dActivation(afunc);

    layer->W = random_matrix(neurons, inputs); 
    layer->bias = random_matrix(neurons, 1);
    layer->dW = zeros_matrix(neurons, inputs);
    layer->dB = zeros_matrix(neurons, 1);

    if (layer->W.data == NULL || layer->bias.data == NULL || 
            layer->dW.data == NULL || layer->dB.data == NULL ) {
        free_matrix(&layer->W);
        free_matrix(&layer->bias);
        free_matrix(&layer->dW);
        free_matrix(&layer->dB);
        return -1;
    }
    return 0;
}

// TODO: Null check and enum these instead of if elsing a written list
dActivation_fn get_dActivation(activation_fn afunc) {
    if (afunc == sigmoid)
        return d_sigmoid;
    if (afunc == relu) 
        return d_relu;
    return NULL;
}


// Activation functions
void relu(const double *z, double *a, size_t neurons) {
    for (size_t neuron=0; neuron < neurons; ++neuron)
        a[neuron] = z[neuron] < 0.0 ? 0.0 : z[neuron];
}

void sigmoid(const double *z, double *a, size_t neurons) {
    for (size_t neuron=0; neuron<neurons; ++neuron) 
        if (z[neuron] >= 0.0)
            a[neuron] = 1 / ( 1+exp(-z[neuron]) );
        else {
            double e = exp(z[neuron]);
            a[neuron] = e / (1.0 + e);
        }
}

void softmax(const double *z, double *a, size_t neurons) {
    double max_z = z[0];
    double total_sum = 0.0;

    for (size_t neuron=0; neuron<neurons; ++neuron)
        if (z[neuron] > max_z)
            max_z = z[neuron];

    for (size_t neuron=0; neuron<neurons; ++neuron) {
        a[neuron] = exp(z[neuron] - max_z);
        total_sum += a[neuron];
    }

    for (size_t neuron=0; neuron<neurons; ++neuron)
        a[neuron] /= total_sum;
}

double d_relu(double a) {
    return a > 0.0 ? 1.0 : 0.0;
}

double d_sigmoid(double a) {
    return a * (1.0 - a);
}

// TODO: this.
void d_softmax() {}

void cleanup(network_t *nn) {
    if (nn == NULL) {
        fprintf(stderr, "cleanup: Network is set to NULL\n");
        return;
    }

    for (size_t i=0; i<nn->layer_count; ++i) {
        free_matrix(&nn->layers[i].W);
        free_matrix(&nn->layers[i].bias);
        free_matrix(&nn->layers[i].dW);
        free_matrix(&nn->layers[i].dB);
    }

    free(nn->layers);
    nn->layers = NULL;
    nn->layer_count = nn->capacity = 0;
}

void set_seed(network_t *nn, size_t seed) {
    if (nn == NULL) {
        fprintf(stderr, "set_seed: Network is set to NULL\n");
        return;
    }

    nn->seed = seed;
    srand(seed);
}

network_t neural_net(size_t features, double lr, size_t epochs, loss_fn loss) {
    network_t nn = {
    .layer_count      = 0,
    .capacity         = 0,
    .input_features   = features,
    .output_classes   = 0,
    .output_layer_idx = 0,
    .learning_rate    = lr,
    .seed             = (unsigned)time(NULL),
    .epochs           = epochs,
    .layers           = NULL,
    .loss             = loss
    };

    srand(nn.seed); 
    return nn;
}











