#ifndef LOSS_H 
#define LOSS_H 

#include "llnnic_types.h"

/* Built in loss function configurations*/
extern const loss_t BINARY_CROSS_ENTROPY;
extern const loss_t CATEGORICAL_CROSS_ENTROPY;

/*
 * Computes binary cross entropy for a single output binary classification 
 * task, averages across the samples
 * 
 * Params: 
 *   layer: output layer to calculate loss for  
 *   y: the sample labels 
 *
 * Returns: 
 *    the loss for the network
 */
double binary_cross_entropy(layer_t layer, matrix_t y);

/* 
 * Computes categorical cross entropy for a mutliclass classification 
 * task, averages across the samples.
 *
 * Params: 
 *   layer: output layer to calculate loss for  
 *   y: the sample labels 
 * 
 * Returns: 
 *    the loss for the network
 */
double categorical_cross_entropy(layer_t layer, matrix_t y);

/* 
 * Computes the binary cross entropy loss gradient w.r.t the output activations 
 * writing the dL/dA into dA
 *
 * Params: 
 *   layer: output layer to calculate loss for  
 *   y: the sample labels 
 */
void binary_cross_entropy_gradient(layer_t layer, matrix_t y, matrix_t *dA);

/* 
 * Computes the categorical cross entropy loss gradient w.r.t output activations 
 * writing the dL/dA into dA
 *
 * Params: 
 *   layer: output layer to calculate loss for  
 *   y: the sample labels 
 */
void categorical_cross_entropy_gradient(layer_t layer, matrix_t y, matrix_t *dA);


#endif 
