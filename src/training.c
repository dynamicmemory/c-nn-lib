#include "training.h"
#include "llnnic_internal.h"

#include "activation.h"
#include "loss.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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


/*
 * Performs back propagation to calculate the loss and gradient updates needed 
 * for the bias and weights of the network.
 * Starts by calculating the loss w.r.t the output layers activation and the 
 * target values, works its way through each layer calculating the prior layers 
 * dA values all the way to the input layer. Averages out all losses over the 
 * input batch size
 *
 * Params:
 *   nn - the neural network whose layer gradients will be calculated
 *   x - input samples
 *   y - target labels for the x samples
 *
 * Returns:
 *   0 on success or -1 on memory allocation failure
*/
int backprop(network_t *nn, matrix_t x, matrix_t y);


/* Updates the networks weights and bias using the accumulated gradients after 
 * backprop has finished. 
 * 
 * Params: 
 *   nn - Network containing the parameterr, grads, and learning rate
 * 
 */
void update_weights_bias(network_t *nn);



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
    matrix_t dZ = {0}, dA = {0};
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
