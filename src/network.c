#include "network.h"

#include "activation.h"
#include "matrix.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
    .threads          = 1,
    .layers           = NULL,
    .loss             = loss 
    };

    srand(nn.seed); 
    return nn;
}

void set_seed(network_t *nn, size_t seed) {
    if (nn == NULL) {
        fprintf(stderr, "set_seed: Network is set to NULL\n");
        return;
    }

    nn->seed = seed;
    srand(seed);
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

void network_cleanup(network_t *nn) {
    if (nn == NULL) {
        fprintf(stderr, "cleanup: Network is set to NULL\n");
        return;
    }

    for (size_t i=0; i<nn->layer_count; ++i) {
        free_matrix(&nn->layers[i].W);
        free_matrix(&nn->layers[i].bias);
        free_matrix(&nn->layers[i].a); // TODO: May cause trouble
        free_matrix(&nn->layers[i].dW);
        free_matrix(&nn->layers[i].dB);
    }

    free(nn->layers);
    nn->layers = NULL;
    nn->layer_count = nn->capacity = 0;
}
