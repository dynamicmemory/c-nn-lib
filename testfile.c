/* TODO: PREDICT: Make it work, for binary and multiclass, consider regression 
          for the future too.*/
// TODO: Onehotencoding for labels... decide if always must and how to handle user
// TODO: Enum for activation / deactivation version to not pass in decactive 
// TODO: weight decay regularization param for a
// TODO: Fix consts throughout the project
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// activation, derivative of activation
typedef void (*activation_fn)(const double *z, double *a, size_t n);
typedef double (*dActivation_fn)(double a);

typedef struct {
    size_t rows;
    size_t cols;
    size_t size;
    double *data;
} matrix_t;

typedef struct {
    size_t inputs;
    size_t neurons;

    matrix_t W;
    matrix_t bias;

    matrix_t a;

    matrix_t dW;
    matrix_t dB;
    activation_fn activation;
    dActivation_fn d_activation;
} layer_t;

typedef double (*loss_fn)(layer_t, matrix_t);

typedef struct {
    size_t layer_count;
    size_t capacity;

    size_t input_features;
    size_t output_classes;

    size_t input_layer; // 0;
    size_t output_layer_idx; // l_count - 1;

    double learning_rate;
    size_t epochs;
    size_t seed;

    loss_fn loss;
    layer_t *layers;
} network_t;

typedef struct {
    size_t class;
    double *probabilities;
} predict_t;

void neural_net(network_t *nn, size_t features, double lr, size_t epochs, loss_fn loss);
void add_layer(network_t *nn, size_t neurons, activation_fn afunc);
layer_t layer(size_t inputs, size_t neurons, activation_fn afunc);
void cleanup(network_t *nn, matrix_t x, matrix_t y);

matrix_t random_matrix(size_t inputs, size_t neurons);
matrix_t zeros_matrix(size_t inputs, size_t neurons);

void relu(const double *z, double *a, size_t n);
void sigmoid(const double *z, double *a, size_t n);
void softmax(const double *z, double *a, size_t n);
double d_relu(double a);
double d_sigmoid(double a);
void d_softmax();
dActivation_fn get_dActivation(activation_fn afunc);

double binary_cross_entropy(layer_t layer, matrix_t y);
double categorical_cross_entropy(layer_t layer, matrix_t y);

void forward(layer_t *layer, matrix_t x);
void backprop(network_t *nn, matrix_t x, matrix_t y);
void update_weights_bias(network_t *nn);
void allocate_activations(layer_t *layer, matrix_t inputs);
void free_activations(network_t *nn);

void set_seed(network_t *nn, double seed);
void train(network_t *nn, matrix_t x, matrix_t y);
void evaluate(network_t *nn, matrix_t x, matrix_t y);
void binary_evaluator(network_t *nn, matrix_t x, matrix_t y);
void predict(network_t *nn, matrix_t m, double *preds);

void binary_test();
void multiclass_test();
matrix_t create_toy_data();
matrix_t create_binary_labels();
matrix_t create_multiclass_labels();

//**************mnist specifically***********************
#include <stdint.h>
static int read_uint32(FILE *file, uint32_t *out);
matrix_t load_mnist_images(const char *path);
matrix_t load_mnist_labels(const char *path);

int mnist_get_label(const matrix_t *labels, size_t index);
void mnist_print_image(const matrix_t *images, size_t index);
void mnist_print_sample(const matrix_t *images, const matrix_t *labels, size_t index);
//**************mnist specifically***********************


int main(int argc, char *argv[]) {
    // binary_test();
    // multiclass_test();
    char mnist_images[] = "./data/mnist/train-images-idx3-ubyte";
    char mnist_labels[] = "./data/mnist/train-labels-idx1-ubyte";
    char test_images[] = "./data/mnist/t10k-images-idx3-ubyte";
    char test_labels[] = "./data/mnist/t10k-labels-idx1-ubyte";
    
    matrix_t x_train = load_mnist_images(mnist_images);
    matrix_t y_train = load_mnist_labels(mnist_labels);

    matrix_t x_test = load_mnist_images(test_images);
    matrix_t y_test = load_mnist_labels(test_labels);
    
    printf("x_train Shape: %zu x %zu\n", x_train.rows, x_train.cols);
    printf("y_train Shape: %zu x %zu\n", y_train.rows, y_train.cols);
    printf("First pixel: %f\n", x_train.data[0]);

    if (x_train.rows == y_train.rows) 
        printf("x.rows == y.rows\n");

    // mnist_print_sample(&x_train, &y_train, 1990);
    // Create a subset to test with 
    int n = 2000;
    matrix_t x_small = {
        .rows = n,
        .cols = x_train.cols,
        .size = n * x_train.cols,
        .data = malloc(n * x_train.cols * sizeof(double))
    };
    matrix_t y_small = {
        .rows = n,
        .cols = y_train.cols,
        .size = n * y_train.cols,
        .data = malloc(n * y_train.cols * sizeof(double))
    };
    for (size_t i=0; i<x_small.size; ++i)
        x_small.data[i] = x_train.data[i];
    for (size_t i=0; i<y_small.size; ++i)
        y_small.data[i] = y_train.data[i];


    network_t nn;
    neural_net(&nn, x_small.cols, 0.2, 500, categorical_cross_entropy);
    set_seed(&nn, 69);
    add_layer(&nn, 32, relu);
    add_layer(&nn, y_small.cols, softmax);

    train(&nn, x_small, y_small);
    printf("Train set ");
    evaluate(&nn, x_small, y_small);
    printf("Test set ");
    evaluate(&nn, x_test, y_test);

    free(x_train.data);
    free(y_train.data);
    free(x_small.data);
    free(y_small.data);
    return 0;
}

// TODO: dw & db can be freed in train, consider resetting the network here too!
void cleanup(network_t *nn, matrix_t x, matrix_t y) {
    for (size_t i=0; i<nn->layer_count; ++i) {
        free(nn->layers[i].W.data);
        free(nn->layers[i].bias.data);
        free(nn->layers[i].dW.data);
        free(nn->layers[i].dB.data);
    }
    free(x.data);
    free(y.data);
}

void set_seed(network_t *nn, double seed) {
    nn->seed = seed;
    srand(seed);
}

void neural_net(network_t *nn, size_t features, double lr, size_t epochs, loss_fn loss) {
    nn->layer_count      = 0;
    nn->capacity         = 0;
    nn->input_features   = features;
    nn->output_classes   = 0;
    nn->output_layer_idx = 0;
    nn->learning_rate    = lr;
    nn->seed             = (unsigned)time(NULL);
    nn->epochs           = epochs;
    nn->layers           = NULL;
    nn->loss             = loss;

    srand(nn->seed); 
}

// TODO: think of defence against non initialized network using this function
void add_layer(network_t *nn, size_t neurons, activation_fn afunc) {
    if (nn->layer_count == nn->capacity)
        nn->capacity = nn->capacity == 0 ? 2 : nn->capacity * 2;

    // TODO: Add failure check
    nn->layers = realloc(nn->layers, nn->capacity * sizeof(layer_t));

    // Get the input size for the layer from prior layers output or input features
    int input_size = nn->input_features;
    size_t i;
    for (i=0; i<nn->layer_count; ++i)
        input_size = nn->layers[i].neurons;

    // Add layer
    layer_t l = layer(input_size, neurons, afunc);
    // TODO: Enum that auto assigns d_activation depending on activation passed in
    nn->layers[i] = l;
    nn->layer_count++;
    nn->output_classes = neurons;
    nn->output_layer_idx = nn->layer_count - 1;
}

// TODO: Null check and enum these instead of if elsing a written list
dActivation_fn get_dActivation(activation_fn afunc) {
    if (afunc == sigmoid)
        return d_sigmoid;
    if (afunc == relu) 
        return d_relu;
    return NULL;
}

layer_t layer(size_t inputs, size_t neurons, activation_fn afunc) {
    layer_t l = { 
        .inputs = inputs,     
        .neurons = neurons, 
        .W = random_matrix(inputs, neurons), 
        .bias = random_matrix(1, neurons),
        .a = {0},
        .dW = zeros_matrix(inputs, neurons),
        .dB = zeros_matrix(1, neurons),
        .activation = afunc,
        .d_activation = get_dActivation(afunc)
    };
    return l;
}

// Activation functions
void relu(const double *z, double *a, size_t neurons) {
    for (size_t neuron=0; neuron < neurons; ++neuron)
        a[neuron] = z[neuron] < 0.0 ? 0.0 : z[neuron];
}

void sigmoid(const double *z, double *a, size_t neurons) {
    for (size_t neuron=0; neuron<neurons; ++neuron) 
        a[neuron] = 1 / ( 1+exp(-z[neuron]) );
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

double d_relu(double a) {
    return a > 0.0 ? 1.0 : 0.0;
}

double d_sigmoid(double a) {
    return a * (1.0 - a);
}

// TODO: this.
void d_softmax() {}

// Loss functions
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

matrix_t random_matrix(size_t inputs, size_t neurons) {
    matrix_t w = {
        .rows = neurons, 
        .cols = inputs,
        .data = malloc(neurons * inputs * sizeof(double))
    };

    for (size_t i=0; i<w.rows*w.cols; ++i)
        w.data[i] = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
    return w;
}

matrix_t zeros_matrix(size_t inputs, size_t neurons) {
    matrix_t w = {
        .rows = neurons, 
        .cols = inputs,
        .data = calloc(neurons * inputs, sizeof(double))
    };
    return w;
}

void forward(layer_t *layer, matrix_t x) {
    double *Z = calloc(layer->neurons, sizeof(double));

    for (size_t row=0; row<x.rows; ++row) {
        for (size_t neuron=0; neuron<layer->neurons; ++neuron) {
            size_t a_idx = row * layer->neurons + neuron;
            double z = layer->bias.data[neuron];

            for (size_t input=0; input<layer->inputs; ++input) {
                size_t w_idx = neuron * layer->W.cols + input;
                size_t x_idx = row * x.cols + input;

                z += layer->W.data[w_idx] * x.data[x_idx]; 
            }
            Z[neuron] = z;
        }
        layer->activation(Z, &layer->a.data[row * layer->neurons], layer->neurons); 
    }
    free(Z);
}

void backprop(network_t *nn, matrix_t x, matrix_t y) {
    layer_t *layer = nn->layers;

    // Reset gradients 
    for (size_t i=0; i<nn->layer_count; ++i) {
        memset(layer[i].dB.data, 0, layer[i].dB.rows * layer[i].dB.cols * sizeof(double));
        memset(layer[i].dW.data, 0, layer[i].dW.rows * layer[i].dW.cols * sizeof(double));
    }

    matrix_t dZ = {0}, dA = {0};
    for (size_t i=nn->layer_count; i-->0; ) {
        dZ.rows = x.rows;
        dZ.cols = nn->layers[i].neurons;
        dZ.data = malloc(dZ.rows * dZ.cols * sizeof(double));

        // Calculating dZ: if output layer, else hidden/input
        if (i == nn->output_layer_idx)
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                    size_t dZ_idx = row * layer[i].neurons + neuron;
                    // TODO: Assumption - bin/categorical cross entropy + sigmoid/softmax
                    dZ.data[dZ_idx] = layer[i].a.data[dZ_idx] - y.data[dZ_idx];
                }
        else {
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                    size_t dZ_idx = row * layer[i].neurons + neuron;
                    size_t dA_idx = row * dA.cols + neuron;
                    dZ.data[dZ_idx] = dA.data[dA_idx] * layer[i].d_activation(layer[i].a.data[dZ_idx]); 
                }
            free(dA.data);
        }

        // Calculate dW: Input layer, else hidden/output
        matrix_t *input_ptr = (i==0) ? &x : &layer[i-1].a;
        for (size_t row=0; row<input_ptr->rows; ++row)
            for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
                layer[i].dB.data[neuron] += dZ.data[row * layer[i].neurons + neuron];

                for (size_t input=0; input<layer[i].inputs; ++input) {
                    size_t w_idx = neuron * layer[i].W.cols + input;
                    size_t a_idx = row * input_ptr->cols + input;
                    size_t dZ_idx = row * layer[i].neurons + neuron;

                    layer[i].dW.data[w_idx] += input_ptr->data[a_idx] * dZ.data[dZ_idx];
                }
            }

        // Calculate dA for current row, for use in the next layers dZ calculations
        if (i > 0) {
            dA.rows=x.rows;
            dA.cols=layer[i].W.cols;
            dA.data= calloc(dA.rows * dA.cols, sizeof(double));
            
            for (size_t row=0; row<x.rows; ++row)
                for (size_t neuron=0; neuron<layer[i].neurons; ++neuron)
                    for (size_t input=0; input<layer[i].inputs; ++input) {
                        size_t w_idx = neuron * layer[i].W.cols + input;
                        size_t dA_idx = row * dA.cols + input;
                        size_t dZ_idx = row * layer[i].neurons + neuron;

                        dA.data[dA_idx] += dZ.data[dZ_idx] * layer[i].W.data[w_idx];
                    }
        }
        free(dZ.data);
    }

    // TODO: Average the gradient losses to match our current loss function evaluation
    for (size_t i=0; i<nn->layer_count; ++i) {
        for (size_t j = 0; j< layer[i].dW.cols * layer[i].dW.rows; ++j)
            layer[i].dW.data[j] /= x.rows;
        for (size_t j = 0; j< layer[i].dB.cols * layer[i].dB.rows; ++j)
            layer[i].dB.data[j] /= x.rows;
    }
}

void update_weights_bias(network_t *nn) {
    layer_t *layer = nn->layers;
    double lr = nn->learning_rate;

    for (size_t i=0; i<nn->layer_count; ++i)
        for (size_t neuron=0; neuron<layer[i].neurons; ++neuron) {
            layer[i].bias.data[neuron] -= lr * layer[i].dB.data[neuron];
            for (size_t input=0; input<layer[i].inputs; ++input) {
                size_t w_idx = neuron * layer[i].W.cols + input;
                layer[i].W.data[w_idx] -= lr * layer[i].dW.data[w_idx];
            }
        }
}

void train(network_t *nn, matrix_t x, matrix_t y){
    double prior_loss = INFINITY;
    double loss, delta, change;
    double epsilon = 1e-6;

    layer_t *layer = nn->layers;
    int count = 0; // TODO: Count be a network variable for a user to call
    while (nn->epochs-- > 0) {
        matrix_t *input = &x;
        for (size_t i=0; i<nn->layer_count; ++i) {
            allocate_activations(&layer[i], *input);
            forward(&layer[i], *input);
            input = &layer[i].a;
        }

        loss = nn->loss(layer[nn->output_layer_idx], y);
        backprop(nn, x, y);
        update_weights_bias(nn);

        delta = loss-prior_loss;
        change = delta < 0 ? delta * -1 : delta;
        if (change < epsilon)
            break;
        prior_loss = loss;

        // TODO: This needs to be user controlled
        // if (count % 500 == 0)
            printf("Epoch: %d | Loss: %f\n", count, loss);
        count++;
    }
    free_activations(nn);
}

void allocate_activations(layer_t *layer, matrix_t inputs) {
    if (layer->a.data != NULL)
        return;

    layer->a.rows = inputs.rows;
    layer->a.cols = layer->neurons;
    layer->a.data = malloc(layer->a.rows * layer->a.cols * sizeof(double));
}

void free_activations(network_t *nn) {
    layer_t *layer = nn->layers;

    for (size_t i=0; i<nn->layer_count; ++i) {
        free(layer[i].a.data);
        layer[i].a.data = NULL;
    }
}

// TODO: x y size match
void binary_evaluator(network_t *nn, matrix_t x, matrix_t y) {
    double *preds = malloc(x.rows * y.cols * sizeof(double));
    predict(nn, x, preds);
    size_t pred = 0;

    for (size_t row=0; row<y.rows; ++row) { 
        size_t best_y = 0;

        pred = preds[row] > 0.5 ? 1 : 0;
        best_y = y.data[row];

        printf("a[%zu]: %f ", row, preds[row]);
    printf("Target: %zu | Predicted: %zu\n", best_y, pred);
    }
    // TODO: As it stands we must free preds here, but this will change so a user 
    // can use the returned values or we will store conf mat + metrics somewhere prior to freeing here 
    free(preds);
}

void evaluate(network_t *nn, matrix_t x, matrix_t y) {
    double *preds = malloc(x.rows * y.cols * sizeof(double));
    predict(nn, x, preds);

    // TODO: Temp accuracy measure 
    double accuracy, pos, neg;
    accuracy = pos = neg = 0;

    for (int row=0; row<y.rows; ++row) { 
        size_t best = 0;
        size_t best_y = 0;

        for (size_t c=0; c<nn->output_classes; ++c) {

            size_t current_class_idx = row * nn->output_classes + c;
            size_t best_class_idx = row * nn->output_classes + best;
            size_t current_target_class = row * y.cols + c;
            size_t predicted_target_class = row * y.cols + best_y;

            if (preds[current_class_idx] > preds[best_class_idx])
                best = c;
            if (y.data[current_target_class] != 0) 
                best_y = c;

            // printf("a[%d]: %f ", row, preds[current_class_idx]);
        }
        // printf("Target: %zu | Predicted: %zu\n", best_y, best);
        if (best_y == best) 
            pos++;
        else 
            neg++;
    }
    printf("Accuracy: %f\n", pos/(pos+neg));
    // TODO: As it stands we must free preds here, but this will change so a user 
    //       can use the returned values or we will store conf mat + metrics somewhere prior to freeing here 
    free(preds);
}

void predict(network_t *nn, matrix_t m, double *preds) {
    layer_t *layer = nn->layers;
    matrix_t *input = &m;

    for (size_t i=0; i<nn->layer_count; ++i) {
        allocate_activations(&layer[i], *input);
        forward(&layer[i], *input);
        input = &layer[i].a;
    }
    for (size_t i=0; i<m.rows * nn->output_classes; ++i)
        preds[i] = layer[nn->output_layer_idx].a.data[i];

    free_activations(nn);
}

//******************************************************************************
// TESTS THE NETWORK 
//******************************************************************************

// TODO: Only for testing 
matrix_t create_toy_data() {
    matrix_t m;
    m.rows=6;
    m.cols=6;
    m.data = calloc(m.rows*m.cols, sizeof(double));
    double d[] = {
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
    add_layer(&nn, 4, relu);
    add_layer(&nn, 4, relu);
    add_layer(&nn, 1, sigmoid);

    train(&nn, x, y);
    binary_evaluator(&nn, x, y);

    // TODO: Consider an easier way for a user to construct predictions
    double new_x[] = {2,1,1,2,3,1};
    matrix_t m = {.rows = 1, .cols = features, .data = new_x};
    double preds[1];
    predict(&nn, m, preds);

    size_t best = 0;
    for (size_t i=0; i<nn.output_classes; ++i)
        if (preds[i] > best)
            best = i;
    printf("For 2,1,1,2,3,1, network predicted: %d\n", preds[best] > 0.5 ? 1 : 0);

    cleanup(&nn, x, y);
}

void multiclass_test() {
    printf("MULTICLASS\n");
    // Create matrix full of data 
    matrix_t x = create_toy_data();
    matrix_t y = create_multiclass_labels();
    size_t features = x.cols;

    network_t nn;

    neural_net(&nn, features, 0.01, 10000, categorical_cross_entropy);
    add_layer(&nn, 10, relu);
    add_layer(&nn, 10, relu);
    add_layer(&nn, 5, relu);
    add_layer(&nn, 3, softmax);

    train(&nn, x, y);
    evaluate(&nn, x, y);

    // TODO: Consider an easier way for a user to construct predictions
    double new_x[] = {5,5,5,6,7,4};
    matrix_t m = {.rows = 1, .cols = features, .data = new_x};
    double preds[nn.output_classes];
    predict(&nn, m, preds);

    size_t best = 0;
    for (size_t i=0; i<m.rows * nn.output_classes; ++i) {
        if (preds[i] > preds[best])
            best = i;
        printf("preds[%zu]: %f, best idx: %zu | val %f\n", i, preds[i], best, preds[best]);
    }
    printf("For 5,5,5,6,7,4 network predicted: %zu\n", best);

    cleanup(&nn, x, y);
    printf("Script end\n");
}


//*********************mnist specifically*****************************
static int read_uint32(FILE *file, uint32_t *out) {
    unsigned char bytes[4];

    if (fread(bytes, 1, 4, file) != 4)
        return 0;

    *out = ((uint32_t)bytes[0]<<24) | ((uint32_t)bytes[1]<<16) | 
           ((uint32_t)bytes[2]<<8)  | (uint32_t)bytes[3];

    return 1;
}

matrix_t load_mnist_images(const char *path) {
    matrix_t x = {0};

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open image file: %s\n", path);
        return x;
    }

    uint32_t magic, samples, rows, cols;
    if ( !read_uint32(file, &magic) || !read_uint32(file, &samples) ||
         !read_uint32(file, &rows)  || !read_uint32(file, &cols) ) { 
        fprintf(stderr, "Failed to read image header\n");
        fclose(file);
        return x;
    }

    if (magic!=2051 || samples==0 || rows==0 || cols==0) {
        fprintf(stderr, "Invalid MNIST image header\n");
        fclose(file);
        return x;
    }

    size_t total_pixels = (size_t)samples * rows * cols;
    unsigned char *pixels = malloc(total_pixels * sizeof(unsigned char));

    if (pixels == NULL) {
        printf("Failed to allocate pixel memory\n");
        fclose(file);
        return x;
    }

    size_t pixels_read = fread(pixels, sizeof(unsigned char), total_pixels, file);

    if (pixels_read != total_pixels) {
        printf("failed to read image pixels\n");
        free(pixels);
        fclose(file);
        return x;
    }

    x.rows = samples;
    x.cols = rows*cols;
    x.size = total_pixels;
    x.data = malloc((size_t)samples*rows*cols*sizeof(double));

    if (x.data == NULL) {
        printf("Failed to allocate matrix\n");
        free(pixels);
        fclose(file);
        return x;
    }
    
    for (size_t i=0; i<x.size; ++i) 
        x.data[i] = pixels[i] / 255.0;
    free(pixels);
    fclose(file);
    return x;
}

matrix_t load_mnist_labels(const char *path) {
    matrix_t y = {0};

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open label file: %s\n", path);
        return y;
    }

    uint32_t magic, n_labels;
    if ( !read_uint32(file, &magic) || !read_uint32(file, &n_labels) ) {
        printf("Failed to read label headers\n");
        fclose(file);
        return y;
    }

    if (magic!=2049 || n_labels==0) {
        fprintf(stderr, "Invalid MNIST label header\n");
        fclose(file);
        return y;
    }

    unsigned char *labels = malloc((size_t)n_labels * sizeof(*labels));

    if (labels == NULL) {
        printf("Failed to allocate label memory\n");
        fclose(file);
        return y;
    }

    size_t labels_read = fread(labels, sizeof(unsigned char), n_labels, file);

    if (labels_read != n_labels) {
        printf("Failed to read all labels\n");
        free(labels);
        fclose(file);
        return y;
    }

    y.rows = n_labels;
    y.cols = 10;
    y.size = (size_t)n_labels * 10;
    y.data = calloc((size_t)n_labels * 10, sizeof(double));

    if (y.data == NULL) {
        printf("Failed to allocate matrix\n");
        free(labels);
        fclose(file);
        return y;
    }

    for (size_t i=0; i<n_labels; ++i) 
        y.data[i*y.cols + labels[i]] = 1.0;

    free(labels);
    fclose(file);
    return y;
}

int mnist_get_label(const matrix_t *labels, size_t index) {
    if (labels == NULL || labels->data == NULL || 
        index >= labels->rows || labels->cols != 10) {
        return -1;
    }

    for (size_t col=0; col<labels->cols; ++col) 
        if (labels->data[index * labels->cols + col] == 1.0) 
            return (int)col;

    return -1;
}

void mnist_print_image(const matrix_t *images, size_t index) {
    if (images == NULL || images->data == NULL || 
        index >= images->rows || images->cols != 784) {
        fprintf(stderr, "Invalid MNIST image or index\n");
        return;
    }

    for (size_t row=0; row<28; ++row) {
        for (size_t col=0; col<28; ++col) {
            double pixel = images->data[index*images->cols + row*28 + col];

            if (pixel < 0.2)      printf(" ");
            else if (pixel < 0.4) printf(".");
            else if (pixel < 0.6) printf("*");
            else if (pixel <0.8)  printf("o");
            else                  printf("#");
        }
        printf("\n");
    }
}

void mnist_print_sample(const matrix_t *images, const matrix_t *labels, size_t index) {
    if (images == NULL || images->data == NULL || 
        labels == NULL || labels->data == NULL || 
        index >= images->rows || index >= labels->rows) {
        fprintf(stderr, "Invalid MNIST sample or index\n");
        return;
    }

    int label = mnist_get_label(labels, index);
    if (label < 0) {
        fprintf(stderr, "mnist_print_sample: Failed to retrieve sample label\n");
        return;
    }

    printf("Sample: %zu\n", index);
    printf("True label: %d\n\n", label);

    mnist_print_image(images, index);
}
