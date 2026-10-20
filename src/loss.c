#include "loss.h"

#include <math.h>
// Loss confgurations
const loss_t BINARY_CROSS_ENTROPY = {
    .value = binary_cross_entropy,
    .gradient = binary_cross_entropy_gradient,
};

const loss_t CATEGORICAL_CROSS_ENTROPY = {
    .value = categorical_cross_entropy,
    .gradient = categorical_cross_entropy_gradient,
};

// Loss functions
double binary_cross_entropy(layer_t layer, matrix_t y) {
    double loss = 0.0;
    double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row) {  
        double a = layer.a.data[row];
        if (a < epsilon) 
            a = epsilon;
        if (a > 1.0 - epsilon) 
            a = 1.0 - epsilon;

        loss -= y.data[row] * log(a) + (1.0 - y.data[row]) * log(1.0 - a); 
    }
    return loss / y.rows;
}

void binary_cross_entropy_gradient(layer_t layer, matrix_t y, matrix_t *dA) {
    const double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row) {
        double a = layer.a.data[row];
        if (a < epsilon) 
            a = epsilon;
        if (a > 1.0 - epsilon) 
            a = 1.0 - epsilon;

        dA->data[row] = -(y.data[row] / a - (1.0 - y.data[row]) / (1.0 - a));
    }
}

double categorical_cross_entropy(layer_t layer, matrix_t y) {
    double loss = 0.0;
    const double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row)
        for (size_t neuron=0; neuron<layer.neurons; ++neuron) {
            size_t idx = row * layer.neurons + neuron;

            if (y.data[idx] != 0.0) {
                double a = layer.a.data[idx];
                if (a < epsilon) a = epsilon;

                loss -= y.data[idx] * log(a);
            }
        }
    return loss /y.rows;
}

void categorical_cross_entropy_gradient(layer_t layer, matrix_t y, matrix_t *dA) {
    const double epsilon = 1e-15;

    for (size_t row=0; row<y.rows; ++row) {
        for (size_t neuron=0; neuron<layer.neurons; ++neuron) {
            size_t idx = row * layer.neurons + neuron;
            double a = layer.a.data[idx];
            if (a < epsilon) 
                a = epsilon;

            dA->data[idx] = -y.data[idx] / a;
        }
    }
}

