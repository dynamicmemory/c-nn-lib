#include <assert.h>
#include <stdio.h>

#include "network.h"
#include "activation.h"
#include "loss.h"

static void test_network_initialisation(void)
{
    network_t nn = neural_net(4, 0.1, 100, CATEGORICAL_CROSS_ENTROPY);

    assert(nn.layers == NULL);
    assert(nn.layer_count == 0);
    assert(nn.capacity == 0);
    assert(nn.input_features == 4);
    assert(nn.learning_rate == 0.1);
    assert(nn.epochs == 100);
    assert(nn.loss.value == categorical_cross_entropy);
    assert(nn.loss.gradient == categorical_cross_entropy_gradient);

    network_cleanup(&nn);
}

static void test_add_layer(void)
{
    network_t nn = neural_net(4, 0.1, 10, CATEGORICAL_CROSS_ENTROPY);

    assert(add_layer(&nn, 5, relu) == 0);
    assert(nn.layer_count == 1);

    layer_t *layer = &nn.layers[0];

    assert(layer->inputs == 4);
    assert(layer->neurons == 5);

    assert(layer->W.rows == 5);
    assert(layer->W.cols == 4);
    assert(layer->W.data != NULL);

    assert(layer->bias.rows == 5);
    assert(layer->bias.cols == 1);
    assert(layer->bias.data != NULL);

    assert(layer->dW.rows == 5);
    assert(layer->dW.cols == 4);
    assert(layer->dW.data != NULL);

    assert(layer->dB.rows == 5);
    assert(layer->dB.cols == 1);
    assert(layer->dB.data != NULL);

    assert(layer->activation == relu);
    assert(layer->d_activation == d_relu);

    network_cleanup(&nn);
}

static void test_multiple_layers(void)
{
    network_t nn = neural_net(4, 0.1, 10, CATEGORICAL_CROSS_ENTROPY);

    assert(add_layer(&nn, 8, relu) == 0);
    assert(add_layer(&nn, 6, sigmoid) == 0);
    assert(add_layer(&nn, 3, softmax) == 0);

    assert(nn.layer_count == 3);

    assert(nn.layers[0].inputs == 4);
    assert(nn.layers[0].neurons == 8);

    assert(nn.layers[1].inputs == 8);
    assert(nn.layers[1].neurons == 6);

    assert(nn.layers[2].inputs == 6);
    assert(nn.layers[2].neurons == 3);

    assert(nn.output_layer_idx == 2);
    assert(nn.output_classes == 3);
    assert(nn.max_neurons == 8);

    network_cleanup(&nn);
}

static void test_capacity_growth(void)
{
    network_t nn = neural_net(2, 0.1, 10, CATEGORICAL_CROSS_ENTROPY);

    /* Add more than the initial capacity of two layers. */
    for (size_t i = 0; i < 5; ++i)
        assert(add_layer(&nn, 3, relu) == 0);

    assert(nn.layer_count == 5);
    assert(nn.capacity >= nn.layer_count);

    network_cleanup(&nn);
}

static void test_repeated_cleanup(void)
{
    network_t nn = neural_net(3, 0.1, 10, CATEGORICAL_CROSS_ENTROPY);

    assert(add_layer(&nn, 4, relu) == 0);

    network_cleanup(&nn);
    network_cleanup(&nn);

    assert(nn.layers == NULL);
    assert(nn.layer_count == 0);
    assert(nn.capacity == 0);
}

int main(void)
{
    test_network_initialisation();
    test_add_layer();
    test_multiple_layers();
    test_capacity_growth();
    test_repeated_cleanup();

    puts("All network tests passed.");
    return 0;
}
