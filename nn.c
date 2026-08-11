//TODO: Randomize weights, bias
#include <stdio.h>
#include <math.h>

#define INPUT_FEATURES 5
#define HIDDEN_NEURONS 3
#define OUTPUT_NEURONS 1
#define DATA_ROWS 3

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

// Extremely verbose forward pass while i figure structure out
void forward(int rows, int neurons, int input_size, 
        double linear_combination[rows][neurons], double weights[neurons][input_size], 
        double inputs[rows][input_size], double activations[rows][neurons], 
        double bias[neurons], double(*func)(double a)) {
    
    for (int row=0; row<rows; ++row)

        for (int neuron=0; neuron<neurons; ++neuron) {
            linear_combination[row][neuron] = bias[neuron];

            for (int input=0; input<input_size; ++input)
                linear_combination[row][neuron] += weights[neuron][input] * inputs[row][input];

            // activations[row][neuron] = 1/(1+exp(-linear_combination[row][neuron]));
            activations[row][neuron] = func(linear_combination[row][neuron]);
        }
}

int main(void) {
    // Test problem, 5 features, input layer
    double X[3][5] = { 
        {1,  2, 4, 2, 1},
        {10, 9, 5, 9, 10},
        {6,  5, 5, 5, 6}
    };
    double labels[] = {0, 1, 1};

    // FORWARD PASS
    // Hard code weights and bias for input layer 
    double W1[HIDDEN_NEURONS][INPUT_FEATURES] = {{0.001, 0.008, 0.002, 0.005, 0.07},
                                {0.001, 0.002, 0.003, 0.004, 0.006},
                                {0.001, 0.002, 0.003, 0.004, 0.006}};
    // Hard code hidden layer weights and bias
    double z1[DATA_ROWS][HIDDEN_NEURONS] = {0.0};
    double a1[DATA_ROWS][HIDDEN_NEURONS] = {0.0};
    double b1[HIDDEN_NEURONS] = {0.009, 0.008, 0.007};

    double W2[OUTPUT_NEURONS][HIDDEN_NEURONS] = {0.003, 0.005, 0.09};
    double z2[DATA_ROWS][OUTPUT_NEURONS] = {0.0};
    double a2[DATA_ROWS][OUTPUT_NEURONS] = {0.0};
    double b2[OUTPUT_NEURONS] = {0.08};

    forward(DATA_ROWS, HIDDEN_NEURONS, INPUT_FEATURES, z1, W1, X, a1, b1, relu); 
    forward(DATA_ROWS, OUTPUT_NEURONS, HIDDEN_NEURONS, z2, W2, a1, a2, b2, sigmoid); 

    double lr = 0.01;

    double dW1[HIDDEN_NEURONS][INPUT_FEATURES] = {{0.0}};
    double db1[HIDDEN_NEURONS] = {0.01};

    double dW2[OUTPUT_NEURONS][HIDDEN_NEURONS] = {{0.0}};
    double db2[OUTPUT_NEURONS] = {0.01};

    // And now we do loss and backprop....

    return 0;
}
    //
    // // Batch foreward pass input layer into first hidden layer 1
    // for (int row=0; row < DATA_ROWS; ++row) {
    //
    //     for (int i=0; i<HIDDEN_NEURONS; ++i) {
    //
    //         z1[row][i] = b1[i];
    //
    //         for (int j=0; j<INPUT_FEATURES; ++j) {
    //
    //             z1[row][i] += W1[i][j] * X[row][j];
    //         }
    //
    //         a1[row][i] = 1 / (1 + exp(-z1[row][i]));       // sigmoid, try relu soon
    //     }
    // }
    //
    // // Batch forward pass hidden layer 1 into output layer
    // for (int row=0; row < DATA_ROWS; ++row) {
    //
    //     for (int i=0; i<OUTPUT_NEURONS; ++i) {
    //
    //         z2[row][i] = b2[i];
    //
    //         for (int j=0; j<HIDDEN_NEURONS; ++j) {
    //
    //             z2[row][i] += W2[i][j] * a1[row][j];
    //         }
    //
    //         a2[row][i] = 1 / (1 + exp(-z2[row][i]));           // sigmoi
    //
    //     }
    // }
    //
    // // Print the hidden activations for the layers
    // for (int i=0; i<DATA_ROWS; ++i) {
    //     printf("a1 row%d ", i);
    //     for (int j=0; j<HIDDEN_NEURONS; ++j) {
    //         printf("%lf ", a1[i][j]);
    //     }
    //     printf("\n");
    // }
    // for (int i=0; i<DATA_ROWS; ++i) {
    //     printf("a2 row%d ", i);
    //     for (int j=0; j<OUTPUT_NEURONS; ++j) {
    //     printf("%lf ", a2[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
