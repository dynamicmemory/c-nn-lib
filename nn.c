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

// rows is the number of rows of the data, 
// neurons is the number of neurons for this layer 
// input_size is the number of neurons/features from previous layer or input 
// linear_combination & weights is self explanatory 
// inputs are either the prior layers activations or the input features 
// actvations and bias are self explain, 
// func is the activation function for the layer, 
// Ill change the names eventually, keeping verbose for my sanity
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

    // input -> w1 -> z1 -> a1 -> w2 -> z2 -> a2 -> loss
    double lr = 0.001;
    int epochs = 5000;

    double dz1[DATA_ROWS][HIDDEN_NEURONS] = {0.0};
    double dW1[HIDDEN_NEURONS][INPUT_FEATURES] = {{0.0}};
    double dA1[DATA_ROWS][HIDDEN_NEURONS] = {0.0};
    double db1[HIDDEN_NEURONS] = {0.01};

    double dz2[DATA_ROWS][OUTPUT_NEURONS] = {0.0};
    double dW2[OUTPUT_NEURONS][HIDDEN_NEURONS] = {{0.0}};
    double db2[OUTPUT_NEURONS] = {0.01};

     // loss 
    double loss[DATA_ROWS][OUTPUT_NEURONS] = {0.0};

    int count = 0;
    while (epochs-- > 0) {

        forward(DATA_ROWS, HIDDEN_NEURONS, INPUT_FEATURES, z1, W1, X, a1, b1, sigmoid); 
        forward(DATA_ROWS, OUTPUT_NEURONS, HIDDEN_NEURONS, z2, W2, a1, a2, b2, sigmoid); 

        // Find and print loss
        for (int row=0; row<DATA_ROWS; ++row) 
            for (int neuron=0; neuron<OUTPUT_NEURONS; ++neuron)
                loss[row][neuron] = -(labels[row] * log(fmin(fmax(a2[row][neuron], 1e-15), 1.0 - 1e-15)) + 
                        (1-labels[row]) * log(1 - fmin(fmax(a2[row][neuron], 1e-15), 1.0 - 1e-15)));

        double total_loss = 0.0;
        for (int row=0; row<DATA_ROWS; ++row)
            for (int output=0; output<OUTPUT_NEURONS; ++output)
                total_loss += loss[row][output];
        double avg_loss = total_loss / DATA_ROWS;

        count++;
        if (epochs % 10 == 0)
            printf("Epoch %d | Loss: %lf\n", count, avg_loss);


        // Backprop
        // a2 -> z2 
        // Store the loss in dz2 by minusing the label from the output of the 
        // activation of the layer 'a2'. This maybe back to front label - a2?
        for (int row=0; row<DATA_ROWS; ++row) 
            for (int output=0; output<OUTPUT_NEURONS; ++output)
                dz2[row][output] = a2[row][output] - labels[row];

        // z2 -> w2 -> a1 -> z1 -> w1 
        // Accumulate the input (a1 activation from prior layer) by the loss found 
        // above (dz2)
        for (int output=0; output<OUTPUT_NEURONS; ++output)
            for (int neuron=0; neuron<HIDDEN_NEURONS; ++neuron)
                for (int row=0; row<DATA_ROWS; ++row) 
                    dW2[output][neuron] += a1[row][neuron] * dz2[row][output];

        // b2
        // get the loss for the bias terms, just assume * 1 therefore just the 
        // error is applied
        for (int output=0; output<OUTPUT_NEURONS; ++output)
            for (int row=0; row<DATA_ROWS; ++row) 
                db2[output] += dz2[row][output];

        // w2 -> a1 -> z1 -> w1 
        for (int row=0; row<DATA_ROWS; ++row)
            for (int neuron=0; neuron<HIDDEN_NEURONS; ++neuron)
                for (int output=0; output<OUTPUT_NEURONS; ++output)
                    dA1[row][neuron] += dz2[row][output] * W2[output][neuron];

        // a1 -> z1 -> w1 
        for (int row=0; row<DATA_ROWS; ++row)
            for (int neuron=0; neuron<HIDDEN_NEURONS; ++neuron)
                dz1[row][neuron] = dA1[row][neuron] * a1[row][neuron] * (1.0 - a1[row][neuron]);

        // z1 -> w1 
        for (int output=0; output<HIDDEN_NEURONS; ++output)
            for (int input=0; input<INPUT_FEATURES; ++input)
                for (int row=0; row<DATA_ROWS; ++row)
                    dW1[output][input] += X[row][input] * dz1[row][output];

        // b1
        for (int output=0; output<HIDDEN_NEURONS; ++output)
            for (int row=0; row<DATA_ROWS; ++row)
                db1[output] += dz1[row][output];

        // update everything 
        for (int output=0; output<OUTPUT_NEURONS; ++output) {
            b2[output] -= lr * db2[output];
            for (int neuron=0; neuron<HIDDEN_NEURONS; ++neuron)
                W2[output][neuron] -= lr * dW2[output][neuron];
        }

        for (int neuron=0; neuron<HIDDEN_NEURONS; ++neuron) {
            b1[neuron] -= lr * db1[neuron];
            for (int feat=0; feat<INPUT_FEATURES; ++feat)
                W1[neuron][feat] -= lr * dW1[neuron][feat];
        }

        // TODO: Reset all d arrays to 0 or else im accumulating grads each epoch
    }


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
