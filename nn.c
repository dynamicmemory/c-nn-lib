//TODO: Randomize weights, bias
#include <stdio.h>
#include <math.h>

#define INPUT 5
#define HIDDEN 3

typedef struct { 
    int neurons;
    int *bias;
    int *activations;
    int *w;
} layer;

// Activation functions
double relu(double z);
double sigmoid(double z);
double softmax();

// Loss functions 
double mse();
double binary_cross_entropy(double t, double y, int size);
double categorical_cross_entropy();

// Optimizers 
double sgd();
double adam();



// Activation functions
double sigmoid(double z) {
    return 1 / (1 + exp(-z));
}

double relu(double z) {
    return z <= 0 ? 0 : z;
}

// Cost/Loss functions / Objective function / Util function
double binary_cross_entropy(double t, double y, int size) {
    // return pow(y, t) * pow(1-y, 1-t);
    return -(t * log(y) + (1-t) * log(1 - y));
}

void backprop(double w[], double x[], double y, double t, int size, double lr, double *bias) {
    double error = y - t;

    for (int i=0; i<size; ++i) {
        w[i] -= lr * error * x[i];
    }
    *bias -= lr * error;
}

int main(void) {
    // Test problem, 5 features, input layer
    double x[3][5] = { 
        {1,  2, 4, 2, 1},
        {10, 9, 5, 9, 10},
        {6,  5, 5, 5, 6}
    };
    double labels[] = {0, 1, 1};

    // Hard code weights and bias for input layer 
    double w1[INPUT] = {0.001, 0.008, 0.002, 0.005, 0.07};

    // Hard code hidden layer weights and bias
    double w2[HIDDEN] = {0.008, 0.005, 0.07};
    double b2[HIDDEN] = {0.009, 0.008, 0.007};

    double b3 = 0.001;

    double lr = 0.01;

    double z[3] = {0.0};
    double y[3] = {0.0};
    double loss[3] = {0.0};
    
    int counter = 1;
    int stopping = 50;
    int epochs = 3;
    while (counter < epochs) {

    }
    return 0;
}

// Activity function
// TODO: Add bias terms
// n - number of neurons, f - activation function, size - number of features,  
// z - preactivation, y - activations, w - weights, x - data, 
void forward(int neurons, double (*activation)(double a), int input_size, 
           double w[][input_size], double x[], double y[], double z[]) {

    // Store the preactivation values for backprop
    // For each neuron
    for (int i=0; i<neurons; ++i) {
        // For each input (feature or neuron from prior layer)
        for (int j=0; j<input_size; ++j) {
            z[i] += w[i][j] * x[j];
        }
    }

    // Perform activation and store values
    for (int i=0; i<neurons; ++i) 
        y[i] = activation(z[i]);
}



