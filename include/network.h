#ifndef NETWORK_H 
#define NETWORK_H

#include "nnlib_types.h"

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
void network_cleanup(network_t *nn);

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

#endif
