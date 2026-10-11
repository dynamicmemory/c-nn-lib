// TODO: Network typedef that creates, assigns, etc the layers
// TODO: Fix consts throughout the project
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
    void (*activation)(const double *z, double *a, size_t n);
    double (*d_activation)(double a);
} layer_t;

// TODO: weight decay regularization param for a
typedef struct {
    size_t l_count;
    size_t capacity;

    size_t input_features;
    size_t output_classes;

    size_t input_layer; // 0;
    size_t output_layer; // l_count - 1;

    double learning_rate;
    size_t epochs;

    double (*loss_function)(layer_t l, matrix_t y);
    layer_t *layers;
} network_t;

void binary_test();
matrix_t create_toy_data();
matrix_t create_binary_labels();
matrix_t create_multiclass_labels();
matrix_t gen_weights_(size_t inputs, size_t neurons, int random);
matrix_t gen_bias_(size_t neurons, int random);
layer_t layer(size_t input, size_t neurons, void (*activation)(const double *z, double *a, size_t n), double(*d_activation)(double));

void relu(const double *z, double *a, size_t n);
void sigmoid(const double *z, double *a, size_t n);
void softmax(const double *z, double *a, size_t n);
double d_relu(double a);
double d_sigmoid(double a);

void d_softmax();
double binary_cross_entropy(layer_t layer, matrix_t y);
double categorical_cross_entropy(layer_t layer, matrix_t y);

void forward(layer_t *layer, matrix_t x);
void backprop(network_t *nn, matrix_t x, matrix_t y);
void train(network_t *nn, matrix_t x, matrix_t y);
void predict(network_t *nn, matrix_t m, double *preds);

void neural_net(network_t *nn, size_t features, double lr, size_t epochs, double (*loss)(layer_t l, matrix_t y)) {
    nn->l_count        = 0;
    nn->capacity       = 0;
    nn->input_features = features;
    nn->output_classes = 0;
    nn->learning_rate  = lr;
    nn->epochs         = epochs;

    nn->layers         = NULL;
    nn->loss_function  = loss;
}

// TODO: think of defence against non initialized network using this function
void add_layer(network_t *nn, size_t neurons, void (*afunc)(const double *z, double *a, size_t n), double(*dfunc)(double a)) {
    if (nn->l_count == nn->capacity)
        nn->capacity = nn->capacity == 0 ? 2 : nn->capacity * 2;

    // TODO: Add failure check
    nn->layers = realloc(nn->layers, nn->capacity * sizeof(layer_t));

    // Get the input size for the layer from prior layers output or input features
    int input_size = nn->input_features;
    size_t i;
    for (i=0; i<nn->l_count; ++i)
        input_size = nn->layers[i].neurons;

    // Add layer
    layer_t l = layer(input_size, neurons, afunc, dfunc);
    // TODO: Enum that auto assigns d_activation depending on activation passed in
    nn->layers[i] = l;
    nn->l_count++;
    nn->output_classes = neurons;
}

int main(int argc, char *argv[]) {
    binary_test();
    //TODO: this will have to be initialized once, not in a main eventually
    // TODO: Random weight initialziation sometimes causes training platue: FIX
    srand((unsigned)time(NULL)); 
    // srand(43); 

    printf("MULTICLASS\n");
    // Create matrix full of data 
    matrix_t x = create_toy_data();
    matrix_t y = create_multiclass_labels();

    int features = x.cols;

    network_t nn;

    neural_net(&nn, features, 0.001, 10000, categorical_cross_entropy);
    add_layer(&nn, 10, relu, d_relu);
    add_layer(&nn, 10, relu, d_relu);
    add_layer(&nn, 5, relu, d_relu);
    add_layer(&nn, 3, softmax, NULL);

    train(&nn, x, y);

    // TODO: Reason, fix, explore
    for (int i=0; i<y.rows; ++i) { 
        size_t best = 0;
        size_t best_y = 0;

        double *aval = nn.layers[nn.l_count-1].a.data;

        for (size_t j=0; j<nn.output_classes; ++j) {
            if (aval[i * nn.output_classes + j] > aval[i * nn.output_classes + best])
                best = j;
            if (y.data[i*y.cols + j] > y.data[i*y.cols + best_y])
                best_y = j;
        }
        printf("Target: %zu | Predicted: %zu\n", best_y, best);
        // printf("Target: %f | Predicted: %ld\n", y.data[i], best);
        printf("\n");
    }

    // TODO: Consider an easier way for a user to construct predictions
    double new_x[] = {2,1,1,2,3,1};
    matrix_t m = {.rows = 1, .cols = features, .data = new_x};
    double preds[1];
    predict(&nn, m, preds);

    size_t best = 0;
    for (size_t i=0; i<nn.output_classes; ++i)
        if (preds[i] > best)
            best = i;
    printf("For 2,1,1,2,3,1, network predicted: %f\n", preds[best]);

    // TODO: Clean up function that generalizes this work
    for (size_t i=0; i<nn.l_count; ++i) {
        free(nn.layers[i].weights.data);
        free(nn.layers[i].bias.data);
        free(nn.layers[i].a.data);
        free(nn.layers[i].dW.data);
        free(nn.layers[i].dB.data);
    }
    free(x.data);
    free(y.data);

    printf("Script end\n");
    return 0;
}

void relu(const double *z, double *a, size_t neurons) {
    for (size_t neuron=0; neuron < neurons; ++neuron)
        a[neuron] = z[neuron] < 0.0 ? 0.0 : z[neuron];
}

double d_relu(double a) {
    return a > 0.0 ? 1.0 : 0.0;
}

void sigmoid(const double *z, double *a, size_t neurons) {
    for (size_t neuron=0; neuron<neurons; ++neuron) 
        a[neuron] = 1 / ( 1+exp(-z[neuron]) );
}

double d_sigmoid(double a) {
    return a * (1.0 - a);
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

// double *d_softmax() {}

layer_t layer(size_t inputs_size, size_t neurons, void (*activation)(const double *z, double *a, size_t n), double(*d_activation)(double)) {
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

void train(network_t *nn, matrix_t x, matrix_t y){
    double prior_loss = INFINITY;
    double loss = 0.0;
    double epsilon = 1e-15;

    int count = 0; // TODO: Count be a network variable for a user to call
    while (nn->epochs-- > 0) {
        matrix_t *input = &x;
        for (size_t i=0; i<nn->l_count; ++i) {
            forward(&nn->layers[i], *input);
            input = &nn->layers[i].a;
        }

        loss = nn->loss_function(nn->layers[nn->l_count-1], y);
        backprop(nn, x, y);

        // TODO: clean up check terminal conditions
        double delta = loss-prior_loss;
        double change = delta < 0 ? delta * -1 : delta;
        if (change < epsilon)
            break;
        prior_loss = loss;

        if (count % 500 == 0)
            printf("Epoch: %d | Loss: %f\n", count, loss);
        count++;
    }
}

void forward(layer_t *layer, matrix_t x) {
    // TODO: Eventually extract/network object to control batch size and activation structure
    if (layer->a.data == NULL) {
        layer->a.rows = x.rows;
        layer->a.cols = layer->neurons;
        layer->a.data = calloc(layer->a.rows*layer->neurons, sizeof(double));
    }
    double *Z = calloc(layer->neurons, sizeof(double));


    for (size_t row=0; row<x.rows; ++row) {
        for (size_t neuron=0; neuron<layer->neurons; ++neuron) {

            size_t a_idx = row * layer->neurons + neuron;
            double z = layer->bias.data[neuron];

            for (size_t input=0; input<layer->inputs; ++input) {

                size_t w_idx = neuron * layer->weights.cols + input;
                size_t x_idx = row * x.cols + input;

                z += layer->weights.data[w_idx] * x.data[x_idx]; 
            }
            // layer->a.data[a_idx] = layer->activation(z); 
            Z[neuron] = z;
        }
        layer->activation(Z, &layer->a.data[row * layer->neurons], layer->neurons); 
    }

    free(Z);
}

void backprop(network_t *nn, matrix_t x, matrix_t y) {
    // TODO: Move the clear gradients from previous training iterations code out
    for (size_t i=nn->l_count; i-- > 0; ) 
        for(size_t row=0; row<nn->layers[i].dW.rows; ++row) {
            nn->layers[i].dB.data[row] = 0;
            for(size_t col=0; col<nn->layers[i].dW.cols; ++col)
                nn->layers[i].dW.data[row * nn->layers[i].dW.cols + col] = 0;
    }
    layer_t *layer = nn->layers;
    matrix_t dZ, dA;
    for (size_t i=nn->l_count; i-- > 0; ) {
        // Create temp matrix to hold dZ
        dZ.rows = x.rows;
        dZ.cols = nn->layers[i].neurons;
        dZ.data =  calloc(x.rows * nn->layers[i].neurons, sizeof(double));

        // Output layer, else hidden/input
        if (i == nn->l_count-1)
            // Works for binary classification using binary cross entropy
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                    size_t dZ_idx = row * layer[i].neurons + neuron;
                    dZ.data[dZ_idx] = layer[i].a.data[dZ_idx] - y.data[dZ_idx];
                }
        else {
            // TODO: Logically shouldnt arrive here without dA assigned, but not impossible!
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                    size_t dZ_idx = row * layer[i].neurons + neuron;
                    size_t dA_idx = row * dA.cols + neuron;
                    dZ.data[dZ_idx] = dA.data[dA_idx] * layer[i].d_activation(layer[i].a.data[dZ_idx]); 
                }

            free(dA.data);
        }

        // TODO: Just assign a point to the x data or dZ data and cut a loop out
        // Input layer, else hidden/output
        if (i == 0)
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                    layer[i].dB.data[neuron] += dZ.data[row * layer[i].neurons + neuron];

                    for (size_t input=0; input<layer[i].inputs; ++input) {

                        size_t w_idx = neuron * layer[i].weights.cols + input;
                        size_t x_idx = row * x.cols + input;
                        size_t dZ_idx = row * layer[i].neurons + neuron;

                        layer[i].dW.data[w_idx] += x.data[x_idx] * dZ.data[dZ_idx];
                    }
                }
        else { 
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                    layer[i].dB.data[neuron] += dZ.data[row * layer[i].neurons + neuron];

                    for (size_t input=0; input<layer[i].inputs; ++input) {

                        size_t w_idx = neuron * layer[i].weights.cols + input;
                        size_t a_idx = row * layer[i-1].a.cols + input;
                        size_t dZ_idx = row * layer[i].neurons + neuron;

                        layer[i].dW.data[w_idx] += layer[i-1].a.data[a_idx] * dZ.data[dZ_idx];
                    }
                }

            // TODO: Calculates when i == 1 which isnt needed, use specific i>1 eventually
            // Assign dA for the next rounds turn of finding dZ
            dA.rows=x.rows;
            dA.cols=layer[i].weights.cols;
            dA.data= calloc(dA.rows * dA.cols, sizeof(double));
            
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron)
                    for (size_t input=0; input<layer[i].inputs; ++input) {
                        size_t w_idx = neuron * layer[i].weights.cols + input;
                        size_t dA_idx = row * dA.cols + input;
                        size_t dZ_idx = row * layer[i].neurons + neuron;

                        dA.data[dA_idx] += dZ.data[dZ_idx] * layer[i].weights.data[w_idx];
                    }
        }
        free(dZ.data);
    }

    // Update all weights and bias
    for (size_t i=0; i<nn->l_count; ++i)
        for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
            layer[i].bias.data[neuron] -= nn->learning_rate * layer[i].dB.data[neuron];
            for (size_t input=0; input<layer[i].inputs; ++input) {
                size_t w_idx = neuron * layer[i].weights.cols + input;
                layer[i].weights.data[w_idx] -= nn->learning_rate * layer[i].dW.data[w_idx];
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

// TODO: Change to output multiclass as well
void predict(network_t *nn, matrix_t m, double *preds) {
    layer_t *layers = nn->layers;
    matrix_t *input = &m;
    for (size_t i=0; i<nn->l_count; ++i) {
        forward(&layers[i], *input);
        input = &layers[i].a;
    }
    // TODO: Sort out this l_count -1 business by storing output layer number
    // double p = layers[nn->l_count-1].a.data[0];
    for (size_t i=0; i<nn->output_classes; ++i)
        preds[i] = layers[nn->l_count-1].a.data[i];
    // return p; 
}

//******************************************************************************

// TODO: Only for testing 
matrix_t create_toy_data() {
    matrix_t m;
    m.rows=6;
    m.cols=6;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {
        // 1,2,3,3,2,1, 
        // 5,6,4,6,4,5, 
        // 4,5,4,3,6,4, 
        // 9,8,9,7,8,6, 
        // 3,2,3,4,2,2,
        // 7,8,7,7,8,6,
        1,2,3,3,2,1, 
        5,5,4,5,4,5, 
        4,5,4,5,5,4, 
        9,8,9,7,8,9, 
        3,2,3,1,2,2,
        7,8,7,7,8,9,
    };
    for (size_t i=0; i<m.rows*m.cols; ++i)
        m.data[i] = d[i];
    return m;
}

// TODO: Only for testing 
matrix_t create_binary_labels() {
    matrix_t m;
    m.rows=6;
    m.cols=1;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {0, 1, 0, 1, 0, 1};
    for (size_t i=0; i<m.rows*m.cols; ++i)
        m.data[i] = d[i];
    return m;
}

matrix_t create_multiclass_labels() {
    matrix_t m;
    m.rows=6;
    m.cols=3;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {
        1, 0, 0,
        0, 1, 0,
        0, 1, 0,
        0, 0, 1,
        1, 0, 0,
        0, 0, 1};
    for (size_t i=0; i<m.rows*m.cols; ++i)
        m.data[i] = d[i];
    return m;
}

void binary_test() {
    printf("BINARY\n");
    // Create matrix full of data 
    matrix_t x = create_toy_data();
    matrix_t y = create_binary_labels();

    int features = x.cols;

    network_t nn;

    neural_net(&nn, features, 0.01, 10000, binary_cross_entropy);
    add_layer(&nn, 4, relu, d_relu);
    add_layer(&nn, 4, relu, d_relu);
    add_layer(&nn, 1, sigmoid, d_sigmoid);

    train(&nn, x, y);

    for (int i=0; i<y.rows; ++i) { 
        size_t best = 0;
        for (size_t j=0; j<nn.output_classes; ++j)
            if (nn.layers[nn.l_count-1].a.data[j] > best)
                best = j;
        printf("Target: %f | Predicted: %f\n", y.data[i], nn.layers[nn.l_count-1].a.data[i*nn.output_classes + best]);
    }

    // TODO: Consider an easier way for a user to construct predictions
    double new_x[] = {2,1,1,2,3,1};
    matrix_t m = {.rows = 1, .cols = features, .data = new_x};
    double preds[1];
    predict(&nn, m, preds);

    size_t best = 0;
    for (size_t i=0; i<nn.output_classes; ++i)
        if (preds[i] > best)
            best = i;
    printf("For 2,1,1,2,3,1, network predicted: %f\n", preds[best]);

    // TODO: Clean up function that generalizes this work
    for (size_t i=0; i<nn.l_count; ++i) {
        free(nn.layers[i].weights.data);
        free(nn.layers[i].bias.data);
        free(nn.layers[i].a.data);
        free(nn.layers[i].dW.data);
        free(nn.layers[i].dB.data);
    }
    free(x.data);
    free(y.data);

}
