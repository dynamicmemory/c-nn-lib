#include "evaluation.h"
#include "llnnic_internal.h"

#include "loss.h"
#include "matrix.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Returns the index of the largest value in an array
 *
 * Params:
 *   vals - point to the array of doubles 
 *   n - number of elements in the array 
 *
 * Returns:
 *   index of largest element
*/
static size_t argmax(const double *values, size_t n);

/*
 * Builds a confusion matrix from target labeles and model predictions. The 
 * confusion matrix must already be allocated and zero initialized. Rows 
 * represent actual classes, columns represent predicted classes.
 *
 * Params:  
 *   y - target labels
 *   preds - model predictions 
 *   eval - evaluation result struct containing threshold and confusion matrix
 *          values 
*/
static void confusion_matrix(matrix_t y, matrix_t preds, evaluate_t *eval);

/*
 * Calculates accuracy, precision, recall and f1 score from a confusion matrix 
 * Binary classificaion calculates positive class precision, recall, and f1 
 * Multiclass classification calculates macro-average precision, recall and f1
 *
 * The matrix must be valid, and the metric fields should be zeroed
 *
 * Params: 
 *   eval - evaluation result containing the confusion matrix and metrics
 */
static void calculate_metrics(evaluate_t *eval, int binary);


// TODO: Change return to a matrix_t object
int predict(network_t *nn, matrix_t x, matrix_t *preds) {
    if (nn == NULL || preds == NULL) 
        return -1;

    layer_t *layer = nn->layers;
    matrix_t *input = &x;
    *preds = (matrix_t){0};

    for (size_t i=0; i<nn->layer_count; ++i) {
        if (allocate_activations(&layer[i], *input) < 0)
            return -1;
        if (forward(&layer[i], *input)) { 
            free_activations(nn);
            return -1;
        }
        input = &layer[i].a;
    }

    layer_t *output = &layer[nn->output_layer_idx];
    *preds = zeros_matrix(x.rows, output->neurons);
    if (preds->data == NULL) {
        free_activations(nn);
        return -1;
    }
    
    // TODO: Assuming this is a possible bug with output classes = 2 for binary but only 1 neuron
    // for (size_t i=0; i<x.rows * nn->output_classes; ++i)
    //     preds->data[i] = output->a.data[i];

    for (size_t i=0; i<preds->size; ++i)
        preds->data[i] = output->a.data[i];

    free_activations(nn);
    return 0;
}

int evaluate(network_t *nn, matrix_t x, matrix_t y, evaluate_t *result, double threshold) {
    if (nn == NULL || result == NULL || x.data == NULL || y.data == NULL ||
        x.rows == 0 || x.rows != y.rows) 
        return -1;

    layer_t *output = &nn->layers[nn->output_layer_idx];
    int binary = output->neurons == 1;

    // Check labels matches network structure
    if ((binary && y.cols != 1) || (!binary && y.cols != output->neurons))
        return -1;

    matrix_t preds = {0};
    evaluate_t eval = {0};
    
    if (predict(nn, x, &preds) != 0)
        return -1;
    
    // Calc loss for the eval set using the loss func for the network
    layer_t loss_layer = *output;
    loss_layer.a = preds;
    eval.loss = nn->loss.value(loss_layer, y);

    if (binary)
        eval.threshold = threshold;
    else 
        eval.threshold = -1.0;

    size_t classes = binary ? 2 : output->neurons; 
    eval.confusion = zeros_matrix(classes, classes);
    if (eval.confusion.data == NULL) {
        free_matrix(&preds);
        return -1;
    }

    confusion_matrix(y, preds, &eval);
    calculate_metrics(&eval, binary);

    free_matrix(&preds);
    *result = eval;
    return 0;
}

static size_t argmax(const double *vals, size_t n) {
    size_t best = 0;
    for (size_t i=1; i<n; ++i)
        if (vals[i] > vals[best])
            best = i;
    return best;
}

static void confusion_matrix(matrix_t y, matrix_t preds, evaluate_t *eval) {
    size_t classes = y.cols == 1 ? 2 : y.cols;

    for (size_t row=0; row<y.rows; ++row) {
        size_t actual = 0;
        size_t predicted = 0;

        // Binary problem
        if (y.cols == 1) {
            actual = y.data[row] >= 0.5 ? 1 : 0;
            predicted = preds.data[row] >= eval->threshold ? 1 : 0;
        }
        // Multiclass problem
        else {
            actual = argmax(&y.data[row*y.cols], y.cols);
            predicted = argmax(&preds.data[row*preds.cols], preds.cols);
        }

        size_t idx = actual*classes + predicted;
        eval->confusion.data[idx] += 1.0;
    }
}

static void calculate_metrics(evaluate_t *eval, int binary) {
    size_t classes = eval->confusion.rows;
    double *cm = eval->confusion.data;

    // Accuracy
    for (size_t i=0; i<classes; ++i)
        eval->accuracy += cm[i*classes + i];

    double total = 0.0;
    for (size_t i=0; i<eval->confusion.size; ++i)
        total += cm[i];

    eval->accuracy = total > 0.0 ? eval->accuracy / total : 0.0;

    // Calculate Binary metrics, recall, precision, f1
    if (binary) {
        double fp = cm[1];
        double fn = cm[2];
        double tp = cm[3];

        // Protect against 0 divide error
        eval->precision = (tp + fp) > 0.0 ? tp / (tp + fp) : 0.0;
        eval->recall = (tp + fn) > 0.0 ? tp / (tp + fn) : 0.0;
        double pr = eval->precision + eval->recall;
        eval->f1_score = pr > 0.0 ? 2.0 * eval->precision * eval->recall / pr : 0.0;

        return;
    }
    // Calculate macro average multiclass metrics, recall, precision, f1
    for (size_t c=0; c<classes; ++c) {
        double tp = cm[c*classes + c];
        double fp = 0.0;
        double fn = 0.0;

        for (size_t other=0; other<classes; ++other)
            if (other != c) {
                fp += eval->confusion.data[other*classes + c];
                fn += eval->confusion.data[c*classes + other];
            }

        double precision = (tp + fp) > 0.0 ? tp / (tp + fp) : 0.0;
        double recall = (tp + fn) > 0.0 ? tp / (tp + fn) : 0.0;
        double pr = precision + recall;
        double f1 = pr > 0.0 ? 2.0 * precision * recall / pr : 0.0;
        eval->precision += precision;
        eval->recall += recall;
        eval->f1_score += f1;
    }
    eval->precision /= (double)classes;
    eval->recall /= (double)classes;
    eval->f1_score /= (double)classes;
}

void print_confusion_matrix(matrix_t confusion) {
    if (confusion.data == NULL)
        return;

    double max = 0.0;
    for (size_t i=0; i < confusion.size; ++i)
        if (confusion.data[i] > max)
            max = confusion.data[i];

    printf("Confusion matrix (rows = actual, columns = predicted):\n");

    printf("%5s", "");
    for (size_t col=0; col < confusion.cols; ++col)
        printf("%10zu", col);
    printf("\n");

    for (size_t row=0; row < confusion.rows; ++row) {
        printf("%9zu", row);
        printf("%c", ' ');

        for (size_t col=0; col < confusion.cols; ++col) {
            double value = confusion.data[row * confusion.cols + col];
            double intensity = max > 0.0 ? sqrt(value / max) : 0.0;
            int r, g, b;

            /* Darker blue for higher counts. */
            r = (int)(240.0 - 180.0 * intensity);
            g = (int)(245.0 - 150.0 * intensity);
            b = 255;

            /* Use white text on darker cells. */
            const char *text = intensity > 0.55 ? "\033[97m" : "\033[30m";

            printf("\033[48;2;%d;%d;%dm%s%9.0f\033[0m ",
                   r, g, b, text, value);
        }
        printf("\n");
    }
}

void print_metrics(const evaluate_t *eval) {
    if (eval == NULL)
        return;

    printf("Loss:      %.6f\n", eval->loss);
    printf("Accuracy:  %.6f\n", eval->accuracy);
    printf("Precision: %.6f\n", eval->precision);
    printf("Recall:    %.6f\n", eval->recall);
    printf("F1 score:  %.6f\n", eval->f1_score);

    if (eval->threshold >= 0.0)
        printf("Threshold: %.3f\n", eval->threshold);
    else
        printf("Decision:  argmax\n");
}

void free_evaluate(evaluate_t *eval) {
    if (eval == NULL) 
        return;

    free_matrix(&eval->confusion);
    *eval = (evaluate_t){0};
}
