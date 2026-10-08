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
    double (*activation)(double);
} layer_t;

matrix_t create_toy_data();
matrix_t create_labels();
matrix_t gen_weights_(size_t inputs, size_t neurons);
matrix_t gen_bias_(size_t neurons);
layer_t layer(size_t input, size_t neurons, double(*activation)(double));

double relu(double a);
double sigmoid(double a);

void forward(layer_t layer, matrix_t x);
void backprop();

void fit();


int main(int argc, char *argv[]) {
    //TODO: this will have to be initialized once, not in a main eventually
    srand((unsigned)time(NULL)); 

    printf("Neural Network\n");
    // Create matrix full of data 
    matrix_t x = create_toy_data();
    matrix_t y = create_labels();
    layer_t l1 = layer(x.cols, 4, relu);
    layer_t l2 = layer(l1.neurons, 4, relu);
    layer_t l3 = layer(l2.neurons, 1, sigmoid);
    
    layer_t layers[] = {l1,l2,l3};

    // Forward 
    l1.a.rows = x.rows;
    l1.a.cols = l1.neurons;
    l1.a.data = calloc(x.rows*l1.neurons, sizeof(double));

    for (size_t row=0; row<x.rows; ++row) {
        for (size_t neuron=0; neuron<l1.neurons; ++neuron) {

            size_t a_idx = row * l1.neurons + neuron;
            double z = l1.bias.data[neuron];

            for (size_t input=0; input<l1.inputs; ++input) {

                size_t w_idx = neuron * l1.neurons + input;
                size_t x_idx = row * x.cols + input;

                z += l1.weights.data[w_idx] * x.data[x_idx]; 

            }
            l1.a.data[a_idx] = l1.activation(z); 
        }
    }

    free(x.data);
    free(y.data);
    for (size_t i=0; i<LAYERS; ++i) {
        free(layers[i].weights.data);
        free(layers[i].bias.data);
        free(layers[i].a.data);
    }
    return 0;
}

matrix_t gen_weights_(size_t inputs, size_t neurons) {
    matrix_t w = {
        .rows = neurons, 
        .cols = inputs,
        .data = calloc(neurons * inputs, sizeof(double))
    };

    for (size_t i=0; i<w.rows*w.cols; ++i)
            w.data[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;

    return w;
}

matrix_t gen_bias_(size_t neurons) {
    matrix_t b = {
        .rows = 1, 
        .cols = neurons,
        .data = calloc(1*neurons, sizeof(double))
    };

    for (size_t i=0; i<b.rows*b.cols; ++i)
            b.data[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;

    return b;
}

double relu(double a) {
    return a < 0 ? 0 : a;
}

double sigmoid(double a) {
    return 1 / ( 1+exp(-a) );
}

layer_t layer(size_t inputs_size, size_t neurons, double(*activation)(double)) {
    layer_t l = { 
        .inputs = inputs_size,     
        .neurons = neurons, 
        .weights = gen_weights_(inputs_size, neurons), 
        .bias = gen_bias_(neurons),
        .a = {0},
        .activation = activation 
    };
    LAYERS++;
    return l;
}

void forward(layer_t layer, matrix_t x) {
    // TODO: Eventually extract/network object to control batch size and activation structure
    if (layer.a.data == NULL) {
        layer.a.rows = x.rows;
        layer.a.cols = layer.neurons;
        layer.a.data = calloc(layer.a.rows*layer.neurons, sizeof(double));
    }

    for (size_t row=0; row<x.rows; ++row) {
        for (size_t neuron=0; neuron<layer.neurons; ++neuron) {

            size_t a_idx = row * layer.neurons + neuron;
            double z = layer.bias.data[neuron];

            for (size_t input=0; input<layer.inputs; ++input) {

                size_t w_idx = neuron * layer.neurons + input;
                size_t x_idx = row * x.cols + input;

                z += layer.weights.data[w_idx] * x.data[x_idx]; 
            }
            layer.a.data[a_idx] = layer.activation(z); 
        }
    }
}

void backprop() {
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

