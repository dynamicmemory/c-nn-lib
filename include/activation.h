#ifndef ACTIVATION_H 
#define ACTIVATION_H 

#include "llnnic_types.h"

/* 
 * Applies the ReLu activation element-wise 
 *
 * Params:
 *   z: Input values before activation
 *   a: output array for the activation
 *   n: number of elements
*/
void relu(const double *z, double *a, size_t n);

/* 
 * Applies the sigmoid activation element-wise 
 *
 * Params:
 *   z: Input values before activation
 *   a: output array for the activation
 *   n: number of elements
*/
void sigmoid(const double *z, double *a, size_t n);

/* 
 * Applies softmax across the input vector to produce a probability distro 
 *
 * Params:
 *   z: Input values before activation
 *   a: output array for probabilities 
 *   n: number of elements
*/
void softmax(const double *z, double *a, size_t n);

/*
 * Computes the ReLu derivative to an incoming gradient 
 *
 * Params:
 *   a: activation values from the forward pass
 *   dA: the loss gradients w.r.t the 'a' values
 *   dZ: the storage for how much the loss changes w.r.t the pre activation values
 *   n: the number of neurons in current sample
 *
 * Returns:
 *  1.0 if a > 0.0, otherwise 0.0
*/
// double d_relu(double a);
void d_relu(const double *a, const double *dA, double *dZ, size_t n);
/*
 * Computes the sigmoid derivative using the activation value
 *
 * Params:
 *   a: activation values from the forward pass
 *   dA: the loss gradients w.r.t the 'a' values
 *   dZ: the storage for how much the loss changes w.r.t the pre activation values
 *   n: the number of neurons in current sample
 *
 * Returns:
 *   the activated value of the input 'a' ~ a * (1.0 - a)
*/
// double d_sigmoid(double a);
void d_sigmoid(const double *a, const double *dA, double *dZ, size_t n);

/*
 * Computes the softmax matrix-vector product using the activation value
 *
 * Params:
 *   a: activation values from the forward pass
 *   dA: the loss gradients w.r.t the 'a' values
 *   dZ: the storage for how much the loss changes w.r.t the pre activation values
 *   n: the number of neurons in current sample
 *
 * Returns:
 *   the activated value of the input 'a' ~ a * (1.0 - a)
*/
// void d_softmax(void);
void d_softmax(const double *a, const double *dA, double *dZ, size_t n);

/**/
dActivation_fn get_dActivation(activation_fn afunc);

#endif 
