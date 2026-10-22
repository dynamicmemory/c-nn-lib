#include "mnist.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Read a 32-bit unsigned interger from a file in big-endian byte order. 
 * 
 * Params:
 *   file - Input file positioned at the next four bytes.
 *   out  - Pointer to receive the decoded integer.
 * 
 * Returns:
 *   1 on success or 0 if the arguments are invalid or fewer than four bytes 
 *   can be read.
 */
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

    if (path == NULL) {
        fprintf(stderr, "load_mnist_images: path is NULL\n");
        return x;
    }

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "load_mnist_images: Failed to open image file: %s\n", path);
        return x;
    }

    uint32_t magic, samples, rows, cols;
    if ( !read_uint32(file, &magic) || !read_uint32(file, &samples) ||
         !read_uint32(file, &rows)  || !read_uint32(file, &cols) ) { 
        fprintf(stderr, "load_mnist_images: Failed to read image header\n");
        fclose(file);
        return x;
    }

    if (magic!=2051 || samples==0 || rows==0 || cols==0) {
        fprintf(stderr, "load_mnist_images: Invalid MNIST image header\n");
        fclose(file);
        return x;
    }

    if ((size_t)samples > SIZE_MAX / rows || 
        (size_t)samples * rows > SIZE_MAX / cols) { 
        fprintf(stderr, "load_mnist_images: Image dimensions overflowed\n");
        fclose(file);
        return x;
    }

    size_t total_pixels = (size_t)samples * rows * cols;
    if (total_pixels > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "load_mnist_images: Matrix size allocation overflowed\n");
        fclose(file);
        return x;
    }

    unsigned char *pixels = malloc(total_pixels * sizeof(unsigned char));
    if (pixels == NULL) {
        printf("load_mnist_images: Failed to allocate pixel memory\n");
        fclose(file);
        return x;
    }

    size_t pixels_read = fread(pixels, sizeof(unsigned char), total_pixels, file);
    if (pixels_read != total_pixels) {
        printf("load_mnist_images: Failed to read image pixels\n");
        free(pixels);
        fclose(file);
        return x;
    }

    x.data = malloc((size_t)samples*rows*cols*sizeof(double));
    if (x.data == NULL) {
        printf("load_mnist_images: Failed to allocate matrix\n");
        free(pixels);
        fclose(file);
        return x;
    }
    x.rows = samples;
    x.cols = rows*cols;
    x.size = total_pixels;
    
    for (size_t i=0; i<x.size; ++i) 
        x.data[i] = pixels[i] / 255.0;
    free(pixels);
    fclose(file);
    return x;
}

matrix_t load_mnist_labels(const char *path) {
    matrix_t y = {0};

    if (path == NULL) {
        fprintf(stderr, "load_mnist_labels: path is NULL\n");
        return y;
    }

    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "load_mnist_labels: Failed to open label file: %s\n", path);
        return y;
    }

    uint32_t magic, n_labels;
    if ( !read_uint32(file, &magic) || !read_uint32(file, &n_labels) ) {
        fprintf(stderr, "load_mnist_labels: Failed to read label headers\n");
        fclose(file);
        return y;
    }

    if (magic!=2049 || n_labels==0) {
        fprintf(stderr, "load_mnist_labels: Invalid MNIST label header\n");
        fclose(file);
        return y;
    }

    unsigned char *labels = malloc((size_t)n_labels * sizeof(*labels));
    if (labels == NULL) {
        fprintf(stderr, "load_mnist_labels: Failed to allocate label memory\n");
        fclose(file);
        return y;
    }

    size_t labels_read = fread(labels, sizeof(unsigned char), n_labels, file);
    if (labels_read != n_labels) {
        fprintf(stderr, "load_mnist_labels: Failed to read all labels\n");
        free(labels);
        fclose(file);
        return y;
    }

    y.rows = n_labels;
    y.cols = 10;
    y.size = (size_t)n_labels * 10;
    y.data = calloc((size_t)n_labels * 10, sizeof(double));

    if (y.data == NULL) {
        fprintf(stderr, "load_mnist_labels: Failed to allocate matrix\n");
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
