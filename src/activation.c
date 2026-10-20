#include "activation.h"

#include <math.h>

void relu(const double *z, double *a, size_t neurons) {
    for (size_t neuron=0; neuron < neurons; ++neuron)
        a[neuron] = z[neuron] < 0.0 ? 0.0 : z[neuron];
}

void sigmoid(const double *z, double *a, size_t neurons) {
    for (size_t neuron=0; neuron<neurons; ++neuron) 
        if (z[neuron] >= 0.0)
            a[neuron] = 1 / ( 1+exp(-z[neuron]) );
        else {
            double e = exp(z[neuron]);
            a[neuron] = e / (1.0 + e);
        }
}

void softmax(const double *z, double *a, size_t neurons) {
    double max_z = z[0];
    double total_sum = 0.0;

    for (size_t neuron=0; neuron<neurons; ++neuron)
        if (z[neuron] > max_z)
            max_z = z[neuron];

    for (size_t neuron=0; neuron<neurons; ++neuron) {
        a[neuron] = exp(z[neuron] - max_z);
        total_sum += a[neuron];
    }

    for (size_t neuron=0; neuron<neurons; ++neuron)
        a[neuron] /= total_sum;
}

void d_relu(const double *a, const double *dA, double *dZ, size_t n){ 
    for (size_t i=0; i<n; ++i) 
        dZ[i] = a[i] > 0.0 ? dA[i] : 0.0;
}

void d_sigmoid(const double *a, const double *dA, double *dZ, size_t n) {
    for (size_t i=0; i<n; ++i) 
        dZ[i] = dA[i] * a[i] * (1.0 - a[i]);
}

void d_softmax(const double *a, const double *dA, double *dZ, size_t n) {
    double dot = 0.0;
    for (size_t i=0; i<n; ++i)
        dot += a[i] * dA[i];

    for (size_t i=0; i<n; ++i) 
        dZ[i] = a[i] * (dA[i] - dot);
}

// TODO: enum these instead of if elsing a written list
dActivation_fn get_dActivation(activation_fn afunc) {
    if (afunc == sigmoid)
        return d_sigmoid;
    if (afunc == relu) 
        return d_relu;
    if (afunc == softmax) 
        return d_softmax;

    return NULL;
}
