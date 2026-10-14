#ifndef MNIST_H
#define MNIST_H 

#include <stddef.h>

typedef struct {
    size_t rows;
    size_t cols;
    size_t size;
    double *data;
} matrix_t;

matrix_t load_mnist_images(const char *path);
matrix_t load_mnist_labels(const char *path);

#endif 

// #include "mnist.h"
#include <stdio.h> 
#include <stdlib.h>
#include <stdint.h>

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

int main(int argc, char *argv[]) {
    char mnist_images[] = "./data/mnist/train-images-idx3-ubyte";
    
    matrix_t x_train = load_mnist_images(mnist_images);
    
    printf("Shape: %zu x %zu\n", x_train.rows, x_train.cols);
    printf("First pixel: %f\n", x_train.data[0]);

    FILE *label_file = fopen("./data/mnist/train-labels-idx1-ubyte", "rb");

    if (label_file == NULL) {
        printf("Failed to open file\n");
        return 1;
    }

    uint32_t magic, num_labs;
    if (read_uint32(label_file, &magic) == 0) {
        printf("Failed to open file\n");
        return 1;
    }

    if (read_uint32(label_file, &num_labs) == 0) {
        printf("Failed to open file\n");
        return 1;
    }
    
    printf("magic number: %d\n", magic);
    printf("Number of labs: %d\n", num_labs);

    unsigned char *labels = malloc(num_labs * sizeof(unsigned char));

    if (labels == NULL) {
        printf("Failed to allocate label memory\n");
        fclose(label_file);
        return 1;
    }

    size_t labels_read = fread(labels, sizeof(unsigned char), num_labs, label_file);

    if (labels_read != num_labs) {
        printf("Failed to read all labels\n");
        free(labels);
        fclose(label_file);
        return 1;
    }

    printf("Labels read: %zu\n", labels_read);
    printf("First label: %u\n", labels[0]);

    matrix_t y = {
        .rows = num_labs,
        .cols = 10, 
        .size = (size_t)num_labs * 10,
        .data = calloc((size_t)num_labs * 10, sizeof(double))
    };

    if (y.data == NULL) {
        printf("Failed to allocate matrix\n");
        free(labels);
        fclose(label_file);
        return 1;
    }

    for (size_t i=0; i<num_labs; ++i) 
        y.data[i*y.cols + labels[i]] = 1.0;

    printf("Shape: %zu x %zu\n", y.rows, y.cols);
    printf("First row:\n");
    for (size_t i=0; i<y.cols; ++i)
        printf("%f ", y.data[i]);

    free(y.data);
    free(labels);
    free(x_train.data);
    fclose(label_file);
    return 0;
}
