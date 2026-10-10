// TODO: Network typedef that creates, assigns, etc the layers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


// TODO: Think of a better way to track layers to free their memory.
static int LAYERS=0;

typedef struct {
    size_t rows;
    size_t cols;
    double *data;
} matrix_t;

typedef struct {
    size_t inputs;
    size_t neurons;

    matrix_t weights;
    matrix_t bias;

    matrix_t a;

    matrix_t dW;
    matrix_t dB;
    double (*activation)(double a);
    double (*d_activation)(double a);
} layer_t;

matrix_t create_toy_data();
matrix_t create_labels();
matrix_t gen_weights_(size_t inputs, size_t neurons, int random);
matrix_t gen_bias_(size_t neurons, int random);
layer_t layer(size_t input, size_t neurons, double(*activation)(double), double(*d_activation)(double));

double relu(double a);
double sigmoid(double a);
double d_relu(double a);
double d_sigmoid(double a);
double binary_cross_entropy(layer_t layer, matrix_t y);

void forward(layer_t *layer, matrix_t x);
void backprop(layer_t *layers, matrix_t x, matrix_t y, double lr);

void backprop(layer_t *layers, matrix_t x, matrix_t y, double lr) {
    // TODO: Move the clear gradients from previous training iterations code out
    for (size_t i=LAYERS; i-- > 0; ) 
        for(size_t row=0; row<layers[i].dW.rows; ++row) {
            layers[i].dB.data[row] = 0;
            for(size_t col=0; col<layers[i].dW.cols; ++col)
                layers[i].dW.data[row * layers[i].dW.cols + col] = 0;
    }

    matrix_t dZ, dA;
    for (size_t i=LAYERS; i-- > 0; ) {
        // Create temp matrix to hold dZ
        dZ.rows = x.rows;
        dZ.cols = layers[i].neurons;
        dZ.data =  calloc(x.rows * layers[i].neurons, sizeof(double));

        // Output layer, else hidden/input
        if (i == LAYERS-1)
            // Works for binary classification using binary cross entropy
            for (size_t row=0; row<x.rows; ++row)
                dZ.data[row] = layers[i].a.data[row] - y.data[row];
        else {
            // TODO: Logically shouldnt arrive here without dA assigned, but not impossible!
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layers[i].neurons; ++neuron) {
                    size_t dZ_idx = row * layers[i].neurons + neuron;
                    size_t dA_idx = row * dA.cols + neuron;
                dZ.data[dZ_idx] = dA.data[dA_idx] * layers[i].d_activation(layers[i].a.data[dZ_idx]); 
                }

            free(dA.data);
        }

        // TODO: Just assign a point to the x data or dZ data and cut a loop out
        // Input layer, else hidden/output
        if (i == 0)
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layers[i].neurons; ++neuron) {

                    layers[i].dB.data[neuron] += dZ.data[row * layers[i].neurons + neuron];

                    for (size_t input=0; input<layers[i].inputs; ++input) {

                        size_t w_idx = neuron * layers[i].weights.cols + input;
                        size_t x_idx = row * x.cols + input;
                        size_t dZ_idx = row * layers[i].neurons + neuron;

                        layers[i].dW.data[w_idx] += x.data[x_idx] * dZ.data[dZ_idx];
                    }
                }
        else { 
            for (size_t row=0; row<x.rows; ++row)

                for (size_t neuron=0; neuron<layers[i].neurons; ++neuron) {

                    layers[i].dB.data[neuron] += dZ.data[row * layers[i].neurons + neuron];

                    for (size_t input=0; input<layers[i].inputs; ++input) {

                        size_t w_idx = neuron * layers[i].weights.cols + input;
                        size_t a_idx = row * layers[i-1].a.cols + input;
                        size_t dZ_idx = row * layers[i].neurons + neuron;

                        layers[i].dW.data[w_idx] += layers[i-1].a.data[a_idx] * dZ.data[dZ_idx];
                    }
                }

            // TODO: Calculates when i == 1 which isnt needed, use specific i>1 eventually
            // Assign dA for the next rounds turn of finding dZ
            dA.rows=x.rows;
            dA.cols=layers[i].weights.cols;
            dA.data= calloc(dA.rows * dA.cols, sizeof(double));
            
            for (size_t row=0; row<x.rows; ++row)

                for (size_t neuron=0; neuron<layers[i].neurons; ++neuron)

                    for (size_t input=0; input<layers[i].inputs; ++input) {
                        size_t w_idx = neuron * layers[i].weights.cols + input;
                        size_t a_idx = row * dA.cols + input;
                        size_t dZ_idx = row * layers[i].neurons + neuron;

                        dA.data[a_idx] += dZ.data[dZ_idx] * layers[i].weights.data[w_idx];
                    }
        }

        free(dZ.data);
    }

    // Update all weights and bias
    for (size_t i=0; i<LAYERS; ++i)
        for (size_t neuron=0; neuron<layers[i].neurons; ++neuron) {
            layers[i].bias.data[neuron] -= lr * layers[i].dB.data[neuron];
            for (size_t input=0; input<layers[i].inputs; ++input) {
                size_t w_idx = neuron * layers[i].weights.cols + input;
                layers[i].weights.data[w_idx] -= lr * layers[i].dW.data[w_idx];
            }
        }
}

double binary_cross_entropy(layer_t layer, matrix_t y) {
    double loss = 0.0;
    double epsilon = 1e-15;
    for (size_t row=0; row<y.rows; ++row) {  
        double a = layer.a.data[row];
        if (a < epsilon) a = epsilon;
        if (a > 1.0 - epsilon) a = 1.0 - epsilon;
        loss -= y.data[row] * log(a) + (1.0 - y.data[row]) * log(1.0 - a); 
    }
    loss /= y.rows;
    return loss;
}

void fit();


int main(int argc, char *argv[]) {
    //TODO: this will have to be initialized once, not in a main eventually
    srand((unsigned)time(NULL)); 

    printf("Neural Network\n");
    // Create matrix full of data 
    matrix_t x = create_toy_data();
    matrix_t y = create_labels();

    int features = x.cols;

    // TODO: Auto assign d_activation depending on activation function
    layer_t l1 = layer(features, 4, relu, d_relu);
    layer_t l2 = layer(l1.neurons, 4, relu, d_relu);
    layer_t l3 = layer(l2.neurons, 1, sigmoid, d_sigmoid);
    
    layer_t layers[] = {l1,l2,l3};

    // "Training" loop
    //TODO: Forward - unify to signal call
    double lr = 0.01;
    int epochs = 10000;
    int count = 0;
    double epsilon = 1e-5;
    double prior_loss = 0.0;
    double loss = 0.0;
    while (epochs-- > 0) {
        matrix_t *input = &x;
        for (size_t i=0; i<LAYERS; ++i) {
            forward(&layers[i], *input);
            input = &layers[i].a;
        }

        loss = binary_cross_entropy(layers[LAYERS-1], y);
        backprop(layers, x, y, 0.01);

        // TODO: clean up check terminal conditions
        double delta = loss-prior_loss;
        double change = delta < 0 ? delta * -1 : delta;
        if (change < epsilon)
            break;
        prior_loss = loss;

        if (count % 50 == 0)
            printf("Epoch: %d | Loss: %f\n", count, loss);
        count++;
    }

    for (int i=0; i<y.rows; ++i) 
        printf("Target: %f | Predicted: %f\n", y.data[i], layers[LAYERS-1].a.data[i]);


    free(x.data);
    free(y.data);
    for (size_t i=0; i<LAYERS; ++i) {
        free(layers[i].weights.data);
        free(layers[i].bias.data);
        free(layers[i].a.data);
        free(layers[i].dW.data);
        free(layers[i].dB.data);
    }
    return 0;
}


double relu(double a) {
    return a < 0.0 ? 0.0 : a;
}

double d_relu(double a) {
    return a > 0.0 ? 1.0 : 0.0;
}

double sigmoid(double a) {
    return 1 / ( 1+exp(-a) );
}

double d_sigmoid(double a) {
    return a * (1.0 - a);
}

// double *softmax() {}
// double *d_softmax() {}

layer_t layer(size_t inputs_size, size_t neurons, double(*activation)(double), double(*d_activation)(double)) {
    layer_t l = { 
        .inputs = inputs_size,     
        .neurons = neurons, 
        .weights = gen_weights_(inputs_size, neurons, 1), 
        .bias = gen_bias_(neurons, 1),
        .a = {0},
        .dW = gen_weights_(inputs_size, neurons, 0),
        .dB = gen_bias_(neurons, 0),
        .activation = activation,
        .d_activation = d_activation
    };
    LAYERS++;
    return l;
}

matrix_t gen_weights_(size_t inputs, size_t neurons, int random) {
    matrix_t w = {
        .rows = neurons, 
        .cols = inputs,
        .data = calloc(neurons * inputs, sizeof(double))
    };

    for (size_t i=0; i<w.rows*w.cols; ++i)
        if (random)
            w.data[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        else 
            w.data[i] = 0;

    return w;
}

matrix_t gen_bias_(size_t neurons, int random) {
    matrix_t b = {
        .rows = 1, 
        .cols = neurons,
        .data = calloc(1*neurons, sizeof(double))
    };

    for (size_t i=0; i<b.rows*b.cols; ++i)
        if (random)
            b.data[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        else 
            b.data[i] = 0;

    return b;
}


void forward(layer_t *layer, matrix_t x) {
    // TODO: Eventually extract/network object to control batch size and activation structure
    if (layer->a.data == NULL) {
        layer->a.rows = x.rows;
        layer->a.cols = layer->neurons;
        layer->a.data = calloc(layer->a.rows*layer->neurons, sizeof(double));
    }

    for (size_t row=0; row<x.rows; ++row) {
        for (size_t neuron=0; neuron<layer->neurons; ++neuron) {

            size_t a_idx = row * layer->neurons + neuron;
            double z = layer->bias.data[neuron];

            for (size_t input=0; input<layer->inputs; ++input) {

                size_t w_idx = neuron * layer->weights.cols + input;
                size_t x_idx = row * x.cols + input;

                z += layer->weights.data[w_idx] * x.data[x_idx]; 
            }
            layer->a.data[a_idx] = layer->activation(z); 
        }
    }
}



//******************************************************************************

// TODO: Only for testing 
matrix_t create_toy_data() {
    matrix_t m;
    m.rows=5;
    m.cols=6;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {1,2,3,3,2,1, 5,6,7,6,9,8, 4,5,4,3,2,4, 9,8,9,7,8,6, 3,2,1,1,2,1};
    for (size_t i=0; i<m.rows*m.cols; ++i)
        m.data[i] = d[i];
    return m;
}

// TODO: Only for testing 
matrix_t create_labels() {
    matrix_t m;
    m.rows=5;
    m.cols=1;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {0, 1, 0, 1, 0};
    for (size_t i=0; i<m.rows*m.cols; ++i)
        m.data[i] = d[i];
    return m;
}

