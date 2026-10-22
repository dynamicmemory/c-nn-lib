#define _GNU_SOURCE
#include <pthread.h>
#include "nnlib_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// PARALLEL TRAIN IMPLEMENTATION
typedef struct thread_args_t thread_args_t;

/*
 * Stores state shard between parallel training workers
 *
 * barrier       - Synchronizes workers between stages of each training epoch. 
 * workers       - Array containing the arguments for all worker threads.
 * worker_count  - Total number of worker threads
 * total_rows    - Total number of samples in the training dataset 
 * prior_loss    - Combined loss from the previous epoch, used for early stopping
 * tol           - Absolute loss-change tolerance for early stopping
 * stop          - Nonzero when workers should stop training 
 * verbose       - Nonzero to print training progress
 */
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

/*
 * Stores the arguments and local state for an individual training worker. 
 * 
 * nn        - Pointer to the main network being trained 
 * local_nn  - Worker-local network containing independent layer buffers 
 * x         - Non-owning view of the workers assigned input samples 
 * y         - Non-owning view of the corresponding target samples 
 * layers    - Array of the workers private layer buffers 
 * id        - Unique index identifying the worker 
 * start_row - Index of the first assigned training sample 
 * end_row   - Index of the last assigned training row exclusive 
 * shared    - pointer to the shared state used to coordinate workers 
 * local_loss - Loss calculated by this worker for its assigned samples 
 */
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



/*
 * Executes the parallel training loop for a single worker thread, this is the 
 * entry point function for the multi-threaded training model, its works by 
 * processing a workers designated samples from the training set into local 
 * private buffers, then running the normal forward and backprop on the 
 * data. A pthread_barrier_wait call is used to synchronize threads with 
 * thread 0 summing all gradients up and adjusting weights accordingly.
 *
 * Params:
 *   arg - Pointer to the workers thread arguments
*/
static void *worker(void *args);

/*
 * Allocates private layer storage and buffers for a parallel thread worker.
 * Copie the networks layer config and allocates independt activation and 
 * gradient buffers to prevent lost updates and race conditions 
 *
 * Params: 
 *   worker - worker whose local layers will be initialized 
 *   samples - Number of samples assigned to the worker 
 */
static int allocate_worker_layers(network_t *nn, layer_t **out, size_t samples_count);

/* Creates a non-owning view of a contigous range of a matrix, the retruned 
 * matrix references the original matrixs data without allocating or coping 
 * it. The returned matrix must not be freed from memory.
 *
 * Params: 
 *   matrix - The source matrix 
 *   start_row - Index of the first row to assign 
 *   end_row - Index of the final row to assign up to but not including 
 *
 * Returns:
 *   A matrix view of the original passed in matrix of a subset of its data 
 */
static matrix_t local_matrix_slice(matrix_t src, size_t start, size_t end);

/* Combines the gradients calculated by all parallel workers, computes a 
 * samples count weighted average of each workers weight and bias gradients and 
 * stores the combined gradients in the main networks graidents
 *
 * Params: 
 *   shared - Shared state type containing the workers and training information
 */
static void reduce_gradients(shared_args_t *worker);

/*
 * Frees the layer storage owned by a parallel worker. Releases the workers 
 * activation and gradient buffers, then frees it local layer array. Does not 
 * free the original network layers, only the temporary private allocations 
 *
 * Params: 
 *   layers - Array of layers to free 
 *   count - Number of layers to free 
 */
static void free_worker_layers(layer_t *layers, size_t count);


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
    if (pthread_barrier_init(&barrier, NULL, thread_count) != 0) {
        free(thread_args); free(tid);
        return -1;
    }

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

        if (allocate_worker_layers(nn, &thread_args[i].layers, end-start) < 0) {
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
static void *worker(void *args) {
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

static int allocate_worker_layers(network_t *nn, layer_t **out, size_t samples) {
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
