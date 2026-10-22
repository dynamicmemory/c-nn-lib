#ifndef MNIST_H
#define MNIST_H 

#include "matrix.h"
#include <stddef.h>

/*
 * Loads the MNIST images from an IDX image file. Each image is flattened into 
 * a single row, with pixel values normalized from 0 255 to the range [0.0, 1.0].
 * 
 * Params: 
 *   path - path to the mnist file
 *
 * Returns: 
 *   A matrix_t with the shape of the number of images (60,000) with 784 cols 
 *   each, or an empty matrix_t on fail.
 */
matrix_t load_mnist_images(const char *path);

/*
 * Loads the MNIST labels from an IDX label file. Labels are represented as 
 * One-Hot encoded rows with 10 columns, corresponding to the digits 0 through 
 * 9.
 *
 * Params: 
 *   path - path to the mnist file
 *    
 * Returns: 
 *   A matrix_t with the shape of the number of labels (60,000) with 10 columns 
 *   corresponding to the digits 0 through 9, or an empty matrix on failure
 */
matrix_t load_mnist_labels(const char *path);

/*
 * Retrieves the digit by a one-hot encoded label row.
 * 
 * Params:  
 *   labels - a matrix_t containing the mnist labels 
 *   index  - the label index to get the class for accepts (0 - 60,000)
 * 
 * 
 */
int mnist_get_label(const matrix_t *labels, size_t index);

/*
 * Print an image as ASCII characters, expects a 28 x 28 image (784 columns).
 * 
 * Params: 
 *   images - a matrix_t containing an mnist training sample
 *   index  - the label index to get the class for accepts (0 - 60,000)
 */
void mnist_print_image(const matrix_t *images, size_t index);

/*
 * Prints an image and its correspoinding digit label. Expects images and labels 
 * to contain the requested samle index 
 *
 * Params: 
 *   images - a matrix_t containing an mnist training sample
 *   labels - a matrix_t containing the mnist labels 
 *   index  - the label index to get the class for accepts (0 - 60,000)
 */
void mnist_print_sample(const matrix_t *images, const matrix_t *labels, size_t index);
#endif 
