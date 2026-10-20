#ifndef LLNNIC_INTERNAL_H
#define LLNNIC_INTERNAL_H

#include "llnnic_types.h"

/* Internal training functions used by evaluation.c */

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

#endif 
