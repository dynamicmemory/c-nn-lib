#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "activation.h"

#define EPSILON 1e-9

static void assert_close(double actual, double expected)
{
    assert(fabs(actual - expected) < EPSILON);
}

static void test_relu(void)
{
    const double z[] = {-2.0, -0.5, 0.0, 0.5, 2.0};
    double a[5];

    relu(z, a, 5);

    assert_close(a[0], 0.0);
    assert_close(a[1], 0.0);
    assert_close(a[2], 0.0);
    assert_close(a[3], 0.5);
    assert_close(a[4], 2.0);
}

static void test_sigmoid(void)
{
    const double z[] = {-2.0, 0.0, 2.0};
    double a[3];

    sigmoid(z, a, 3);

    assert_close(a[0], 1.0 / (1.0 + exp(2.0)));
    assert_close(a[1], 0.5);
    assert_close(a[2], 1.0 / (1.0 + exp(-2.0)));

    for (size_t i = 0; i < 3; ++i)
        assert(a[i] > 0.0 && a[i] < 1.0);
}

static void test_softmax(void)
{
    const double z[] = {1.0, 2.0, 3.0};
    double a[3];

    softmax(z, a, 3);

    const double sum = a[0] + a[1] + a[2];

    assert_close(sum, 1.0);
    assert(a[0] > 0.0);
    assert(a[1] > a[0]);
    assert(a[2] > a[1]);
}

static void test_d_relu(void)
{
    const double a[] = {0.0, 0.0, 0.5, 2.0};
    const double dA[] = {1.0, 2.0, 3.0, 4.0};
    double dZ[4];

    d_relu(a, dA, dZ, 4);

    assert_close(dZ[0], 0.0);
    assert_close(dZ[1], 0.0);
    assert_close(dZ[2], 3.0);
    assert_close(dZ[3], 4.0);
}

static void test_d_sigmoid(void)
{
    const double a[] = {0.0, 0.5, 1.0};
    const double dA[] = {1.0, 2.0, 3.0};
    double dZ[3];

    d_sigmoid(a, dA, dZ, 3);

    assert_close(dZ[0], 0.0);
    assert_close(dZ[1], 0.5);
    assert_close(dZ[2], 0.0);
}

static void test_get_d_activation(void)
{
    assert(get_dActivation(relu) == d_relu);
    assert(get_dActivation(sigmoid) == d_sigmoid);
    assert(get_dActivation(softmax) == d_softmax);
    assert(get_dActivation(NULL) == NULL);
}

int main(void)
{
    test_relu();
    test_sigmoid();
    test_softmax();
    test_d_relu();
    test_d_sigmoid();
    test_get_d_activation();

    puts("All activation tests passed.");
    return 0;
}
