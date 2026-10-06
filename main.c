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
    double (*activation)(double);
} layer_t;

matrix_t create_toy_data();
matrix_t create_labels();
matrix_t gen_weights_(size_t inputs, size_t neurons);
matrix_t gen_bias_(size_t neurons);
layer_t layer(size_t input, size_t neurons, double(*activation)(double));

double relu(double a) {
    return a < 0 ? 0 : 1;
}

double sigmoid(double a) {
    return 1 / ( 1+exp(-a) );
}

void forward() {

}

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


    free(x.data);
    free(y.data);
    for (size_t i=0; i<LAYERS; ++i) {
        free(layers[i].weights.data);
        free(layers[i].bias.data);
    }
    return 0;
}

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

layer_t layer(size_t input, size_t neurons, double(*activation)(double)) {
    layer_t l = { 
        .inputs = input,     
        .neurons = neurons, 
        .weights = gen_weights_(input, neurons), 
        .bias = gen_bias_(neurons),  
        .activation = activation 
    };
    LAYERS++;
    return l;
}

matrix_t gen_weights_(size_t inputs, size_t neurons) {
    matrix_t w = {
        .rows = inputs, 
        .cols = neurons,
        .data = calloc(inputs*neurons, sizeof(double))
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
