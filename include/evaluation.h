#ifndef EVALUATION_H 
#define EVALUATION_H 

#include "llnnic_types.h"

/*
 * Generates predictions for a batch of input samples 1 - n, runs forward prop 
 * and copies the output layer activations into a newly allocated preds matrix 
 * 
 * Params:
 *   nn - network used to gen preds 
 *   x - input matrix, shaped (samples * features)
 *   preds - matrix where all predictions made are stored
 *
 * Returns: 
 *   0 on succes and -1 on failure
*/
int predict(network_t *nn, matrix_t x, matrix_t *preds);

/*
 * Evaluates a model on a labelled dataset. Calculates loss, confusion matrices 
 * accuracy, predision, recall, and f1_score for the given model. Binary 
 * uses threshold for splitting classes, multiclass uses argmax 
 *
 * Params: 
 *   nn - the model to evaluate
 *   x - Input feature matrix 
 *   y - input target labels 
 *   result - recevices the evaluation results, including an allocated 
 *            confusion matrix.
 *   threshold - Binary classificaiotn threshold between 0.0 and 1.0. Set to  
 *               -1.0 for multicalss problems 
 *
 * Returns: 
 *   0 on success and -1 on allocation failure
 */
int evaluate(network_t *nn, matrix_t x, matrix_t y, evaluate_t *result, double threshold);

/*
 * Prints a confusion matrix to stdout
 *
 * Params: 
 *   confusion - matrix_t confusion matrixs to print
 */
void print_confusion_matrix(matrix_t confusion);

/*
 * Prints the evaluation metrix to stdout 
 *
 * Currently supports: 
 * Binary - loss, accuracy, precision, recall, f1 and decision threshold 
 * Multiclass - loss, macro average; accuracy, precidion, recall, and f1
 *
 * Params: 
 *   eval - evaluation_t holding results to print
 */
void print_metrics(const evaluate_t *eval);

/*
 * Releases resources owned by an evlation result and resets it to zero
*/
void free_evaluate(evaluate_t *eval);

#endif 
