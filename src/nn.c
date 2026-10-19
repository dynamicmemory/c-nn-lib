// TODO: Onehotencoding for labels... decide if always must and how to handle user
// TODO: weight decay regularization param for a
// TODO: Fix consts throughout the project
// TODO: setup error code enum to propagate correct errors
// TODO: Consider throwing a validation set into train with a NULL check if you dont want it
// TODO: Multithreaded solution to improve performance
#include "nn.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <string.h>



size_t argmax(const double *vals, size_t n) {
    size_t best = 0;
    for (size_t i=1; i<n; ++i)
        if (vals[i] > vals[best])
            best = i;
    return best;
}


void confusion_matrix(matrix_t y, matrix_t preds, evaluate_t *eval) {
    size_t classes = y.cols == 1 ? 2 : y.cols;

    for (size_t row=0; row<y.rows; ++row) {
        size_t actual = 0;
        size_t predicted = 0;

        // Binary problem
        if (y.cols == 1) {
            actual = y.data[row] >= 0.5 ? 1 : 0;
            predicted = preds.data[row] >= eval->threshold ? 1 : 0;
        }
        // Multiclass problem
        else {
            actual = argmax(&y.data[row*y.cols], y.cols);
            predicted = argmax(&preds.data[row*preds.cols], preds.cols);
        }

        size_t idx = actual*classes + predicted;
        eval->confusion.data[idx] += 1.0;
    }
}

void calculate_metrics(evaluate_t *eval, int binary) {
    size_t classes = eval->confusion.rows;
    double *cm = eval->confusion.data;

    // Accuracy
    for (size_t i=0; i<classes; ++i)
        eval->accuracy += cm[i*classes + i];

    double total = 0.0;
    for (size_t i=0; i<eval->confusion.size; ++i)
        total += cm[i];

    eval->accuracy = total > 0.0 ? eval->accuracy / total : 0.0;

    // Calculate Binary metrics, recall, precision, f1
    if (binary) {
        double fp = cm[1];
        double fn = cm[2];
        double tp = cm[3];

        // Protect against 0 divide error
        eval->precision = (tp + fp) > 0.0 ? tp / (tp + fp) : 0.0;
        eval->recall = (tp + fn) > 0.0 ? tp / (tp + fn) : 0.0;
        double pr = eval->precision + eval->recall;
        eval->f1_score = pr > 0.0 ? 2.0 * eval->precision * eval->recall / pr : 0.0;

        return;
    }
    // Calculate macro average multiclass metrics, recall, precision, f1
    for (size_t c=0; c<classes; ++c) {
        double tp = cm[c*classes + c];
        double fp = 0.0;
        double fn = 0.0;

        for (size_t other=0; other<classes; ++other)
            if (other != c) {
                fp += eval->confusion.data[other*classes + c];
                fn += eval->confusion.data[c*classes + other];
            }

        double precision = (tp + fp) > 0.0 ? tp / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0.0 ? tp / (tp + fn) : 0.0;
        double pr = precision + recall;
        double f1 = pr > 0.0 ? 2.0 * precision * recall / pr : 0.0;
        eval->precision += precision;
        eval->recall += recall;
        eval->f1_score += f1;
    }
    eval->precision /= (double)classes;
    eval->recall /= (double)classes;
    eval->f1_score /= (double)classes;
}

int evaluate(network_t *nn, matrix_t x, matrix_t y, evaluate_t *result, double threshold) {
    if (nn == NULL || result == NULL || x.data == NULL || y.data == NULL ||
        x.rows == 0 || x.rows != y.rows) 
        return -1;

    layer_t *output = &nn->layers[nn->output_layer_idx];
    int binary = output->neurons == 1;

    // Check labels matches network structure
    if ((binary && y.cols != 1) || (!binary && y.cols != output->neurons))
        return -1;

    matrix_t preds = {0};
    evaluate_t eval = {0};
    
    if (predict(nn, x, &preds) != 0)
        return -1;
    
    // Calc loss for the eval set using the loss func for the network
    layer_t loss_layer = *output;
    loss_layer.a = preds;
    eval.loss = nn->loss.value(loss_layer, y);

    if (binary)
        eval.threshold = threshold;
    else 
        eval.threshold = -1.0;

    size_t classes = binary ? 2 : output->neurons; 
    eval.confusion = zeros_matrix(classes, classes);
    if (eval.confusion.data == NULL) {
        free_matrix(&preds);
        return -1;
    }

    confusion_matrix(y, preds, &eval);
    calculate_metrics(&eval, binary);

    free_matrix(&preds);
    *result = eval;
    return 0;
}

void print_confusion_matrix(matrix_t confusion) {
    if (confusion.data == NULL)
        return;

    double max = 0.0;
    for (size_t i=0; i < confusion.size; ++i)
        if (confusion.data[i] > max)
            max = confusion.data[i];

    printf("Confusion matrix (rows = actual, columns = predicted):\n");

    printf("%5s", "");
    for (size_t col=0; col < confusion.cols; ++col)
        printf("%10zu", col);
    printf("\n");

    for (size_t row=0; row < confusion.rows; ++row) {
        printf("%9zu", row);
        printf("%c", ' ');

        for (size_t col=0; col < confusion.cols; ++col) {
            double value = confusion.data[row * confusion.cols + col];
            double intensity = max > 0.0 ? sqrt(value / max) : 0.0;
            int r, g, b;

            /* Darker blue for higher counts. */
            r = (int)(240.0 - 180.0 * intensity);
            g = (int)(245.0 - 150.0 * intensity);
            b = 255;

            /* Use white text on darker cells. */
            const char *text = intensity > 0.55 ? "\033[97m" : "\033[30m";

            printf("\033[48;2;%d;%d;%dm%s%9.0f\033[0m ",
                   r, g, b, text, value);
        }
        printf("\n");
    }
}

void print_metrics(const evaluate_t *eval) {
    if (eval == NULL)
        return;

    printf("Loss:      %.6f\n", eval->loss);
    printf("Accuracy:  %.6f\n", eval->accuracy);
    printf("Precision: %.6f\n", eval->precision);
    printf("Recall:    %.6f\n", eval->recall);
    printf("F1 score:  %.6f\n", eval->f1_score);

    if (eval->threshold >= 0.0)
        printf("Threshold: %.3f\n", eval->threshold);
    else
        printf("Decision:  argmax\n");
}

void free_evaluate(evaluate_t *eval) {
    if (eval == NULL) 
        return;

    free_matrix(&eval->confusion);
    *eval = (evaluate_t){0};
}




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
    
    // TODO: Assuming this is a possible bug with output classes = 2 for binary but only 1 neuron
    // for (size_t i=0; i<x.rows * nn->output_classes; ++i)
    //     preds->data[i] = output->a.data[i];

    for (size_t i=0; i<preds->size; ++i)
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
