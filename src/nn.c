/* TODO: PREDICT: Make it work, for binary and multiclass, consider regression 
          for the future too.*/
// TODO: Onehotencoding for labels... decide if always must and how to handle user
// TODO: weight decay regularization param for a
// TODO: Fix consts throughout the project
// TODO: setup error code enum to propagate correct errors
#include "nn.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>

// TODO: x y size match
void binary_evaluator(network_t *nn, matrix_t x, matrix_t y) {
    matrix_t preds = {0};
    predict(nn, x, &preds);
    size_t pred = 0;

    for (size_t row=0; row<y.rows; ++row) { 
        size_t best_y = 0;

        pred = preds.data[row] > 0.5 ? 1 : 0;
        best_y = y.data[row];

        printf("a[%zu]: %f ", row, preds.data[row]);
    printf("Target: %zu | Predicted: %zu\n", best_y, pred);
    }
    // TODO: As it stands we must free preds here, but this will change so a user 
    // can use the returned values or we will store conf mat + metrics somewhere prior to freeing here 
    free(preds.data);
}


// typedef struct {
//     double loss;
//     double threshold;
//     double accuacy;
//     double f1_socre;
//     double recall;
//     double precision;
//     double *class counts;
//     matrix_t Tp, fn, tn, fp; maybe i dunno
//     matrix_t confusion;
// } evaluate_t;

void evaluate(network_t *nn, matrix_t x, matrix_t y) {
    matrix_t preds = {0};
    predict(nn, x, &preds);

    // TODO: Temp accuracy measure 
    double pos, neg;
    pos = neg = 0;

    for (size_t row=0; row<y.rows; ++row) { 
        size_t best = 0;
        size_t best_y = 0;

        for (size_t c=0; c<nn->output_classes; ++c) {

            size_t current_class_idx = row * nn->output_classes + c;
            size_t best_class_idx = row * nn->output_classes + best;
            size_t current_target_class = row * y.cols + c;
            // size_t predicted_target_class = row * y.cols + best_y;

            if (preds.data[current_class_idx] > preds.data[best_class_idx])
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
    free(preds.data);
}

// double f1_score(evaluate_t eval);
// double precision(evaluate_t eval);
// double accuracy(evaluate_t eval);
// double recall(evaluate_t eval);

// TODO: Change return to a matrix_t object
int predict(network_t *nn, matrix_t x, matrix_t *preds) {
    if (nn == NULL || preds == NULL) 
        return -1;

    layer_t *layer = nn->layers;
    matrix_t *input = &x;
    *preds = (matrix_t){0};

    for (size_t i=0; i<nn->layer_count; ++i) {
        if (allocate_activations(&layer[i], *input) < 0)
            return -1;
        if (forward(&layer[i], *input)) { 
            free_activations(nn);
            return -1;
        }
        input = &layer[i].a;
    }

    layer_t *output = &layer[nn->output_layer_idx];
    *preds = zeros_matrix(x.rows, output->neurons);
    if (preds->data == NULL) {
        free_activations(nn);
        return -1;
    }
    
    for (size_t i=0; i<x.rows * nn->output_classes; ++i)
        preds->data[i] = output->a.data[i];

    free_activations(nn);
    return 0;
}

//***************************************************************************************************************************************
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

    nn->max_neurons = neurons > nn->max_neurons ? neurons : nn->max_neurons;
    return 0;
}

int layer_init(layer_t *layer, size_t inputs, size_t neurons, activation_fn afunc) {
    *layer = (layer_t){0};

    layer->inputs = inputs;
    layer->neurons = neurons; 

    layer->activation = afunc;
    layer->d_activation = get_dActivation(afunc);
    if (layer->activation == NULL || layer->d_activation == NULL) 
        return -1;

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

// TODO: enum these instead of if elsing a written list
dActivation_fn get_dActivation(activation_fn afunc) {
    if (afunc == sigmoid)
        return d_sigmoid;
    if (afunc == relu) 
        return d_relu;
    if (afunc == softmax) 
        return d_softmax;

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

void d_relu(const double *a, const double *dA, double *dZ, size_t n){ 
    for (size_t i=0; i<n; ++i) 
        dZ[i] = a[i] > 0.0 ? dA[i] : 0.0;
}

void d_sigmoid(const double *a, const double *dA, double *dZ, size_t n) {
    for (size_t i=0; i<n; ++i) 
        dZ[i] = dA[i] * a[i] * (1.0 - a[i]);
}

void d_softmax(const double *a, const double *dA, double *dZ, size_t n) {
    double dot = 0.0;
    for (size_t i=0; i<n; ++i)
        dot += a[i] * dA[i];

    for (size_t i=0; i<n; ++i) 
        dZ[i] = a[i] * (dA[i] - dot);
}

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

// network_t neural_net(size_t features, double lr, size_t epochs, loss_fn loss) {
network_t neural_net(size_t features, double lr, size_t epochs, loss_t loss) {
    network_t nn = {
    .layer_count      = 0,
    .capacity         = 0,
    .input_features   = features,
    .output_classes   = 0,
    .output_layer_idx = 0,
    .max_neurons      = 0,
    .learning_rate    = lr,
    .seed             = (unsigned)time(NULL),
    .epochs           = epochs,
    .layers           = NULL,
    .loss             = loss 
    };

    srand(nn.seed); 
    return nn;
}

// Loss confgurations
const loss_t BINARY_CROSS_ENTROPY = {
    .value = binary_cross_entropy,
    .gradient = binary_cross_entropy_gradient,
};

const loss_t CATEGORICAL_CROSS_ENTROPY = {
    .value = categorical_cross_entropy,
    .gradient = categorical_cross_entropy_gradient,
};

// Loss functions
double binary_cross_entropy(layer_t layer, matrix_t y) {
    double loss = 0.0;
    double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row) {  
        double a = layer.a.data[row];
        if (a < epsilon) 
            a = epsilon;
        if (a > 1.0 - epsilon) 
            a = 1.0 - epsilon;

        loss -= y.data[row] * log(a) + (1.0 - y.data[row]) * log(1.0 - a); 
    }
    return loss / y.rows;
}

void binary_cross_entropy_gradient(layer_t layer, matrix_t y, matrix_t *dA) {
    const double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row) {
        double a = layer.a.data[row];
        if (a < epsilon) 
            a = epsilon;
        if (a > 1.0 - epsilon) 
            a = 1.0 - epsilon;

        dA->data[row] = -(y.data[row] / a - (1.0 - y.data[row]) / (1.0 - a));
    }
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

void categorical_cross_entropy_gradient(layer_t layer, matrix_t y, matrix_t *dA) {
    const double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row) {
        for (size_t neuron=0; neuron<layer.neurons; ++neuron) {
            size_t idx = row * layer.neurons + neuron;
            double a = layer.a.data[idx];
            if (a < epsilon) 
                a = epsilon;

            dA->data[idx] = -y.data[idx] / a;
        }
    }
}

/*
 * Internal backprop helper function
 * Reshape a matrices metadata without reallocating its dynamically allocated 
 * buffer 
 *
 * Params: 
 *   dn - Matrix whos rows, cols and size will be updated
 *   rows - Requested number of rows
 *   cols - Requested number of cols
*/ 
static void backprop_adjust_rows_cols(matrix_t *dn, size_t rows, size_t cols);

/*
 * Internal backprop helper function
 * Calculate the gradient w.r.t a layers preactivation values
 *
 * Params: 
 *   layer - layer containing activations and the activation derivative 
 *   dA - Incoming gradient dL/dA with one row per sample
 *   dZ - out going location for the gradient dL/dZ 
*/ 
static void backprop_calculate_dZ(layer_t *layer, const matrix_t *dA, matrix_t *dZ, size_t rows);

/*
 * Internal backprop helper function
 * Accumulates the weight and bias gradients over the batch
 *
 * Params: 
 *   layer - layer containing activations and the activation derivative 
 *   x - Inputs to this layer; either the network input batch or the preceeding  
 *       layers activations depending on the layer passed in.
 *   dZ - Gradient w.r.t this layers pre activation values
*/ 
static void backprop_calculate_dW_dB(layer_t *layer, const matrix_t *x, const matrix_t *dZ);

/*
 * Internal backprop helper function
 * Propagates the gradient w.r.t the current layers pre activation values.
 *
 * Params: 
 *   layer - layer containing activations and the activation derivative 
 *   dZ - Gradient w.r.t this layers pre activation values
 *   dA - Out going buffer for the propagated gradient w.r.t the current layers 
 *        inputs
 *   rows - number of samples to process
*/ 
static void backprop_calculate_dA(layer_t *layer, const matrix_t *dZ, matrix_t *dA, size_t rows);

int forward(layer_t *layer, matrix_t x) {
    double *Z = malloc(layer->neurons * sizeof(double));
    if (Z == NULL) { 
        // fprintf(stderr, "forward: allocating Z matrix failed.\n");
        return -1;
    }

    for (size_t row=0; row<x.rows; ++row) {
        for (size_t neuron=0; neuron<layer->neurons; ++neuron) {
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
    return 0;
}

static void backprop_adjust_rows_cols(matrix_t *dn, size_t rows, size_t cols) {
    dn->rows = rows;
    dn->cols = cols;
    dn->size = rows * cols;
}

static void backprop_calculate_dZ(layer_t *layer, const matrix_t *dA, matrix_t *dZ, size_t rows) {
    for (size_t row=0; row<rows; ++row) {
        size_t dZ_idx = row * layer->neurons;
        size_t dA_idx = row * dA->cols;
        double *ptr_a = &layer->a.data[dZ_idx];
        double *ptr_dA = &dA->data[dA_idx];
        double *ptr_dZ = &dZ->data[dZ_idx];

        layer->d_activation(ptr_a, ptr_dA, ptr_dZ, layer->neurons);
    }
}

static void backprop_calculate_dW_dB(layer_t *layer, const matrix_t *x, const matrix_t *dZ) {
    for (size_t row=0; row<x->rows; ++row)
        for (size_t neuron=0; neuron<layer->neurons; ++neuron) {
            layer->dB.data[neuron] += dZ->data[row * layer->neurons + neuron];

            for (size_t input=0; input<layer->inputs; ++input) {
                size_t w_idx = neuron * layer->W.cols + input;
                size_t a_idx = row * x->cols + input;
                size_t dZ_idx = row * layer->neurons + neuron;

                layer->dW.data[w_idx] += x->data[a_idx] * dZ->data[dZ_idx];
            }
        }
}

static void backprop_calculate_dA(layer_t *layer, const matrix_t *dZ, matrix_t *dA, size_t rows) {
    for (size_t row=0; row<rows; ++row)
        for (size_t neuron=0; neuron<layer->neurons; ++neuron)
            for (size_t input=0; input<layer->inputs; ++input) {
                size_t w_idx = neuron * layer->W.cols + input;
                size_t dA_idx = row * dA->cols + input;
                size_t dZ_idx = row * layer->neurons + neuron;

                dA->data[dA_idx] += dZ->data[dZ_idx] * layer->W.data[w_idx];
            }
}

int backprop(network_t *nn, matrix_t x, matrix_t y) {
    layer_t *layer = nn->layers;

    // Reset gradients 
    for (size_t i=0; i<nn->layer_count; ++i) {
        memset(layer[i].dB.data, 0, layer[i].dB.size * sizeof(double));
        memset(layer[i].dW.data, 0, layer[i].dW.size * sizeof(double));
    }

    // Allocate the dZ and dA buffers once to the maximum needed size
    matrix_t dZ, dA;
    dZ.data = malloc(nn->max_neurons * x.rows * sizeof(double)); 
    dA.data = calloc(nn->max_neurons * x.rows, sizeof(double));
    if (dZ.data == NULL || dA.data == NULL) {
        free(dZ.data); free(dA.data);
        return -1;
    }

    for (size_t i=nn->layer_count; i-->0; ) {
        backprop_adjust_rows_cols(&dZ, x.rows, nn->layers[i].neurons);

        // Calculating dZ: if output layer, else hidden/input
        if (i == nn->output_layer_idx) {
            backprop_adjust_rows_cols(&dA, layer[i].a.rows, layer[i].a.cols);
            nn->loss.gradient(layer[i], y, &dA);
        }
        backprop_calculate_dZ(&layer[i], &dA, &dZ, x.rows);

        // Calculate dW: Input layer, else hidden/output
        matrix_t *input_ptr = (i==0) ? &x : &layer[i-1].a;
        backprop_calculate_dW_dB(&layer[i], input_ptr, &dZ);

        // Calculate dA for current row, for use in the next layers dZ calculations
        if (i > 0) {
            backprop_adjust_rows_cols(&dA, x.rows, layer[i].W.cols);
            memset(dA.data, 0, dA.size * sizeof(double));
            
            backprop_calculate_dA(&layer[i], &dZ, &dA, x.rows);
        }
    }

    // TODO: Average the gradient losses to match our current loss function evaluation
    for (size_t i=0; i<nn->layer_count; ++i) {
        for (size_t j = 0; j< layer[i].dW.size; ++j)
            layer[i].dW.data[j] /= x.rows;
        for (size_t j = 0; j< layer[i].dB.size; ++j)
            layer[i].dB.data[j] /= x.rows;
    }

    free(dZ.data);
    free(dA.data);
    return 0;
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

int train(network_t *nn, matrix_t x, matrix_t y, double tol, int verbose){
    double prior_loss = INFINITY;
    double loss, delta, change;
    layer_t *layer = nn->layers;

    for (size_t epoch=0; epoch<nn->epochs; ++epoch) {
        matrix_t *input = &x;

        for (size_t i=0; i<nn->layer_count; ++i) {

            if (allocate_activations(&layer[i], *input) < 0 || 
                forward(&layer[i], *input) < 0)
                return -1;
            input = &layer[i].a;
        }

        loss = nn->loss.value(layer[nn->output_layer_idx], y);
        if (backprop(nn, x, y) < 0)
                return -1;
        update_weights_bias(nn);

        delta = loss-prior_loss;
        change = delta < 0 ? delta * -1 : delta;
        if (change < tol)
            break;
        prior_loss = loss;

        if (verbose)
            printf("Epoch: %zu | Loss: %f\n", epoch, loss);
    }
    free_activations(nn);
    return 0;
}

int allocate_activations(layer_t *layer, matrix_t inputs) {
    if (layer->a.data != NULL)
        return 0;

    layer->a.rows = inputs.rows;
    layer->a.cols = layer->neurons;
    layer->a.size = layer->a.rows * layer->a.cols;
    layer->a.data = malloc(layer->a.rows * layer->a.cols * sizeof(double));

    if (layer->a.data == NULL) 
        return -1;

    return 0;
}

void free_activations(network_t *nn) {
    layer_t *layer = nn->layers;

    for (size_t i=0; i<nn->layer_count; ++i) {
        free(layer[i].a.data);
        layer[i].a.data = NULL;
    }
}
