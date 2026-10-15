#ifndef MNIST_H
#define MNIST_H 

#include "nn.h"
#include <stddef.h>

/*
 *
 *
 */
matrix_t load_mnist_images(const char *path);
/*
 *
 *
 */
matrix_t load_mnist_labels(const char *path);
/*
 *
 *
 */
int mnist_get_label(const matrix_t *labels, size_t index);
/*
 *
 *
 */
void mnist_print_image(const matrix_t *images, size_t index);
/*
 *
 *
 */
void mnist_print_sample(const matrix_t *images, const matrix_t *labels, size_t index);
#endif 
