#define _GNU_SOURCE

#include "training.h"
#include "llnnic_internal.h"

#include "activation.h"
#include "loss.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#include <stdatomic.h>
#include <stdint.h>

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

int serial_train(network_t *nn, matrix_t x, matrix_t y, double tol, int verbose);
int parallel_train(network_t *nn, matrix_t x, matrix_t y, double tol, int verbose);

int train(network_t *nn, matrix_t x, matrix_t y, double tol, int verbose){
    // Add null checks, etc 

    int e;
    if (nn->threads > 1)
        e = parallel_train(nn, x, y ,tol, verbose);
    else 
        e = serial_train(nn, x, y, tol, verbose);

    // TODO: Could just retrun the function itself? same thing
    return e;
}

// SERIAL TRAIN IMPLEMENTATION
int serial_train(network_t *nn, matrix_t x, matrix_t y, double tol, int verbose) {
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
    layer->a.data = calloc(layer->a.rows * layer->a.cols, sizeof(double));

    if (layer->a.data == NULL) 
        return -1;

    return 0;
}

void free_activations(network_t *nn) {
    layer_t *layer = nn->layers;

    for (size_t i=0; i<nn->layer_count; ++i) {
        free_matrix(&layer[i].a);
    }
}


// PARALLEL TRAIN IMPLEMENTATION
typedef struct thread_args_t thread_args_t;

typedef struct {
    pthread_barrier_t *barrier;

    thread_args_t *workers;
    size_t worker_count;
    size_t total_rows;

    double prior_loss;
    double epoch_loss;
    double tol;

    int stop;
    int verbose;
} shared_args_t;

struct thread_args_t {
    network_t *nn;
    network_t local_nn;

    matrix_t x;
    matrix_t y;
    layer_t *layers;

    size_t id;
    size_t start_row;
    size_t end_row;
    
    shared_args_t *shared;
    double local_loss;
};

void *worker(void *args);
int alloc_work_layers(network_t *nn, layer_t **out, size_t samples_count);
static void free_worker_layers(layer_t *layers, size_t count);
static matrix_t local_matrix_slice(matrix_t src, size_t start, size_t end);
static void reduce_gradients(shared_args_t *worker);

int parallel_train(network_t *nn, matrix_t x, matrix_t y, double tol, int verbose) {
    size_t thread_count = nn->threads;

    pthread_t *tid = calloc(thread_count, sizeof(*tid));
    thread_args_t *thread_args = calloc(thread_count, sizeof(*thread_args));

    if (tid == NULL || thread_args == NULL) {
        free(tid); free(thread_args);
        return -1;
    }

    shared_args_t shared = {0};
    shared.workers = thread_args;
    shared.worker_count = thread_count;
    shared.total_rows = x.rows;
    shared.prior_loss = INFINITY;
    shared.tol = tol;
    shared.verbose = verbose;

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, thread_count);
    shared.barrier = &barrier;

    for (size_t i=0; i<thread_count; ++i) {
        size_t start = i * x.rows / thread_count; 
        size_t end = (i + 1) * x.rows / thread_count; 

        thread_args[i].id = i;
        thread_args[i].nn = nn;
        thread_args[i].shared = &shared;

        thread_args[i].start_row = start;
        thread_args[i].end_row = end;
        thread_args[i].x = local_matrix_slice(x, start, end); 
        thread_args[i].y = local_matrix_slice(y, start, end);

        if (alloc_work_layers(nn, &thread_args[i].layers, end-start) < 0) {
            for (size_t i=0; i<thread_count; ++i)
                free_worker_layers(thread_args[i].layers, nn->layer_count);
            pthread_barrier_destroy(&barrier);
            free(thread_args);
            free(tid);
            return -1;
        }

        thread_args[i].local_nn = *nn;
        thread_args[i].local_nn.layers = thread_args[i].layers;
    }

    for (size_t i=0; i<thread_count; ++i)
        pthread_create(&tid[i], NULL, worker, &thread_args[i]);
    for (size_t i=0; i<thread_count; ++i)
        pthread_join(tid[i], NULL);

    return 0;
}

// TODO: Figure out the error returns here
void *worker(void *args) {
    thread_args_t *targs = args;
    network_t *nn = &targs->local_nn;
    shared_args_t *shared = targs->shared;

    for (size_t epoch=0; epoch < targs->nn->epochs; ++epoch) {
        matrix_t *input = &targs->x;

        for (size_t i=0; i<nn->layer_count; ++i) {
            if (forward(&nn->layers[i], *input) < 0)
                return NULL;
            input = &nn->layers[i].a;
        }
        
        targs->local_loss = nn->loss.value(nn->layers[nn->output_layer_idx], targs->y);
        if (backprop(nn, targs->x, targs->y) < 0) 
            return NULL;

        pthread_barrier_wait(shared->barrier);

        if (targs->id == 0) {
            double loss = 0.0;

            for (size_t t=0; t<shared->worker_count; ++t) {
                thread_args_t *w = &shared->workers[t];

                double weight = (double)w->x.rows / (double)shared->total_rows;
                loss += weight * w->local_loss;
            }

            // targs->epoch_loss = loss;
            shared->epoch_loss = loss;

            reduce_gradients(shared);
            update_weights_bias(targs->nn);

            double change = fabs(loss - shared->prior_loss);
            shared->stop = change < shared->tol;
            shared->prior_loss = loss;

            if (shared->verbose & !shared->stop)
                printf("Epoch: %zu | Loss: %f\n", epoch, loss);
        }

        pthread_barrier_wait(shared->barrier);
        if (shared->stop)
            break;
    }

    pthread_exit(NULL);
}


static matrix_t local_matrix_slice(matrix_t src, size_t start, size_t end) {
    return (matrix_t) {
        .rows = end-start, 
        .cols = src.cols, 
        .size = (end-start) * src.cols, 
        .data = src.data + start * src.cols 
    };
}

int alloc_work_layers(network_t *nn, layer_t **out, size_t samples) {
    layer_t *layers = calloc(nn->layer_count, sizeof(*layers));

    if (layers == NULL) return -1;

    for (size_t i=0; i<nn->layer_count; ++i) {
        layers[i] = nn->layers[i]; 

        // TODO: Could i save on memsetting later by clearing out here? 
        layers[i].a = (matrix_t){0};
        layers[i].dW = (matrix_t){0};
        layers[i].dB = (matrix_t){0};

        // TODO: Changed original code to this, not sure if it will work
        matrix_t inputs = {samples, 0, 0, 0};
        allocate_activations(&layers[i], inputs);

        layers[i].dW = nn->layers[i].dW;
        layers[i].dB = nn->layers[i].dB;
        layers[i].dW.data = calloc(layers[i].dW.size, sizeof(double));
        layers[i].dB.data = calloc(layers[i].dB.size, sizeof(double));

        if (layers[i].a.data == NULL || layers[i].dW.data == NULL || layers[i].dB.data == NULL) {
            for (size_t j=0; j<=i; ++j) {
                free(layers[j].a.data);
                free(layers[j].dW.data);
                free(layers[j].dB.data);
            }
            free(layers);
            return -1;
        }
    }
    *out = layers;
    return 0;
}

static void reduce_gradients(shared_args_t *shared) {
    network_t *nn = shared->workers[0].nn;
    
    for (size_t i=0; i<nn->layer_count; ++i) {
        memset(nn->layers[i].dW.data, 0, nn->layers[i].dW.size * sizeof(double));
        memset(nn->layers[i].dB.data, 0, nn->layers[i].dB.size * sizeof(double));
    }

    for (size_t t=0; t<shared->worker_count; ++t) {
        thread_args_t *worker = &shared->workers[t];
        
        size_t local_rows = worker->x.rows;
        double weight = (double)local_rows / (double)shared->total_rows;
        for (size_t i=0; i<nn->layer_count; ++i) {
            layer_t *dst = &nn->layers[i];
            layer_t *src = &worker->layers[i];

            for (size_t j=0; j<dst->dW.size; ++j)
                dst->dW.data[j] += weight * src->dW.data[j];

            for (size_t j=0; j<dst->dB.size; ++j)
                dst->dB.data[j] += weight * src->dB.data[j];
        }
    }
}

static void free_worker_layers(layer_t *layers, size_t count) {
    if (layers == NULL) 
        return;

    for (size_t i=0; i<count; ++i) {
        free(layers[i].a.data);
        free(layers[i].dW.data);
        free(layers[i].dB.data);
    }

    free(layers);
}

