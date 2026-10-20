#ifndef MATRIX_H 
#define MATRIX_H 

#include "llnnic_types.h"

/*
 * Creates a rows by cols matrix populated with random values
 * 
 * Params: 
 *   rows - number of rows in the matrix 
 *   cols - number of cols in the matrix 
 * 
 * Returns: 
 *   A constructed matrix or an empty matrix if allocation failures or dims 
 *   are invalid
*/
matrix_t random_matrix(size_t rows, size_t cols);

/*
 * Creates a rows by cols matrix initialized with zeros 
 * 
 * Params: 
 *   rows - number of rows in the matrix 
 *   cols - number of cols in the matrix 
 * 
 * Returns: 
 *   A constructed matrix or an empty matrix if allocation failures or dims 
 *   are invalid
*/
matrix_t zeros_matrix(size_t rows, size_t cols);

/*
 * Frees a matrix_t types allocated memory in its .data variable, resets the 
 * matrix to 0/NULL 
 * 
 * Params: 
 *   m - the matrix to free and cleanup the state of. 
*/
void free_matrix(matrix_t *m);

#endif 
