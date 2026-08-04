#include <stdio.h>
#include <math.h>

// Linear transformer
double linear_transform(double weights[], double x[], double bias, int size);

// Activation functions
double sigmoid(double z);

// Loss functions 
double binary_cross_entropy(double t, double y, int size);


double linear_transform(double w[], double x[], double bias, int size) {
    double output = 0.0;
    
    for (int i=0; i<size; ++i)
        output += w[i] * x[i];
    
    return output + bias;
}

// Activation functions
double sigmoid(double z) {
    return 1 / (1 + exp(-z));
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
    double x[3][5] = {
        {1, 2, 5, 2, 1},
        {10, 9, 5, 9, 10},
        {6, 5, 5, 5, 6}
    };
    double t[] = {0, 1, 1};

    double weights[] = {0.001, 0.008, 0.002, 0.005, 0.07};
    double bias = 0.00009;
    double lr = 0.01;

    double z[3] = {0.0};
    double y[3] = {0.0};
    double loss[3] = {0.0};
    
    int counter = 1;
    int stopping = 50;
    while (counter < stopping) {
        printf("Training epoch %d\n", counter);
        for (int i=0; i<3; ++i) {
            // Forward pass
            z[i] = linear_transform(weights, x[i], bias, 5);
            y[i] = sigmoid(z[i]);
            loss[i] = binary_cross_entropy(t[i], y[i], 5);

            // Back prop
            backprop(weights, x[i], y[i], t[i], 5, lr, &bias);
        
            // Eval
            printf("  Row: %d Linear_transform: %.4lf | Sigmoid: %.4lf"
                       "| Error: %.4lf\n", i, z[i], y[i], loss[i]);
        }
        printf("-----------------------------------------------------------------\n");

        // Check it classifies the training data correctly
        if (counter == stopping - 1) {
            for (int i = 0; i < 3; i++)
            {
                double a = linear_transform(weights, x[i], bias, 5);
                double p = sigmoid(a);

                printf("Target=%g Prediction=%f\n", t[i], p);
            }

            // try some predictions
            double new_data[] = {2, 3, 4, 3, 2};
            double a = linear_transform(weights, new_data, bias, 5); 
            double pred = sigmoid(a);
            printf("Target: 0, Pred: %lf\n", pred);

            double new_data2[] = {5, 5, 5, 5, 5};
            a = linear_transform(weights, new_data2, bias, 5); 
            pred = sigmoid(a);
            printf("Target: 1, Pred: %lf\n", pred);

            double new_data3[] = {6, 5, 1, 5, 6};
            a = linear_transform(weights, new_data3, bias, 5); 
            pred = sigmoid(a);
            printf("Target: 1, Pred: %lf\n", pred);
        }
        counter++;
    }

    return 0;
}
