#ifndef NNLIB_INTERNAL_H
#define NNLIB_INTERNAL_H

#include "nnlib_types.h"

/* Internal training functions used by evaluation.c, parallel_train.c, and training.c */

/* 
 * Performs the forward pass for a single layer for the neural net. Calculates 
 * the weighted sum of the inputs plus bias for each neuron, then applies the  
 * activation function. 
 *
 * Params:
 *   layer: Ponter to the layer being evaluated
 *   x:     Input matrix with one sample per row and one feature per column
 *
 * Returns: 
 *   0 on success and -1 if memory failed to allocate for the Z scores
 */
int forward(layer_t *layer, matrix_t x);

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

/*
 * Allocates the activation matrix for a layer based on its input batch 
 * 
 * Params: 
 *   layer - Layer whose activation matrix will be allocated for 
 *   inputs - input matrix used to determine the batch row count
 *    
 * Returns:    
 *    0 on success, -1 on allocation failure
 */
int allocate_activations(layer_t *layer, matrix_t inputs);

/* 
 * Free the activation buffers belonging to all network layers 
 * 
 * Params: 
 *   nn - Network containing layer activations to be freed
 */
void free_activations(network_t *nn);

/*
 * Trains the network using multi-threaded full-batch gradient decent.
 *
 * Params:
 * Params: 
 *   nn - Network to train
 *   x - Input samples, with one sample per row 
 *   y - Target values corresponding to the samples in x (multiclass must be one 
 *       hot encoded, binary can be 0, 1 single column vector)
 *   tol - Absolute loss-change tolerance used as the early-stopping threshold, 
 *         (set to 0 if you want all epochs to run)
 *   verbose - If nonzero, will print epochs and loss during training.
 * 
 * Returns:  
 *   0 on success, or -1 on training failure
 */
int parallel_train(network_t *nn, matrix_t x, matrix_t y, double tol, int verbose);

#endif 
