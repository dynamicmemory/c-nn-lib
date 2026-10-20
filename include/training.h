#ifndef TRAINING_H
#define TRAINING_H 

#include "llnnic_types.h"

/* 
 * Trains a model using the supplied input and target data
 * 
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
int train(network_t *nn, matrix_t x, matrix_t y, double tol, int verbose);


#endif
