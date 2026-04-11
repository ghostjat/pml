#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Pointer macro for fast access
#define F32(tensor) ((float*)(tensor)->data)

// ----------------------------------------------------------------------------
// Internal Helper: Rapidly scans a CSV to determine exact rows and columns
// ----------------------------------------------------------------------------
static void _csv_shape(const char* filepath, int has_header, int* out_rows, int* out_cols) {
    FILE* fp = fopen(filepath, "r");
    if (!fp) { 
        fprintf(stderr, "FATAL [Dataset IO]: Cannot open file %s.\n", filepath); 
        exit(1); 
    }

    char buffer[65536]; // 64KB read buffer for maximum disk throughput
    int rows = 0;
    int cols = 0;

    // First line determines column count
    if (fgets(buffer, sizeof(buffer), fp)) {
        cols = 1;
        for (int i = 0; buffer[i]; i++) {
            if (buffer[i] == ',') cols++;
        }
        if (!has_header) rows++;
    }

    // Fast-forward count remaining rows
    while (fgets(buffer, sizeof(buffer), fp)) {
        rows++;
    }

    fclose(fp);
    *out_rows = rows;
    *out_cols = cols;
}

// ----------------------------------------------------------------------------
// Direct-to-Tensor CSV Parser
// ----------------------------------------------------------------------------
Tensor** tensor_dataset_from_csv(const char* filepath, int label_col, int has_header) {
    int rows = 0, cols = 0;
    _csv_shape(filepath, has_header, &rows, &cols);

    if (rows == 0 || cols == 0) return NULL;

    // 1. Determine Tensor Shapes
    int feat_cols = (label_col >= 0) ? cols - 1 : cols;
    
    // Allocate contiguous memory for Samples (Features)
    Tensor* samples = tensor_create_dtype(2, (int[]){rows, feat_cols}, DTYPE_FLOAT32);
    
    // Allocate memory for Labels (Targets) if requested
    Tensor* labels = NULL;
    if (label_col >= 0) {
        labels = tensor_create_dtype(1, (int[]){rows}, DTYPE_FLOAT32);
    }

    // 2. Stream Data Directly to RAM
    FILE* fp = fopen(filepath, "r");
    char buffer[65536];
    
    if (has_header) fgets(buffer, sizeof(buffer), fp); // Skip header

    int r = 0;
    while (fgets(buffer, sizeof(buffer), fp) && r < rows) {
        char* ptr = buffer;
        char* next;
        int c_feat = 0;

        for (int c = 0; c < cols; c++) {
            // Native C string-to-float (Highly optimized by GCC/Clang)
            float val = strtof(ptr, &next);

            if (c == label_col) {
                F32(labels)[r] = val;
            } else {
                F32(samples)[r * feat_cols + c_feat] = val;
                c_feat++;
            }

            // Move pointer to the next column
            if (*next == ',') ptr = next + 1;
            else ptr = next;
        }
        r++;
    }

    fclose(fp);

    // Return the array of pointers: [Samples, Labels]
    Tensor** out = (Tensor**)malloc(2 * sizeof(Tensor*));
    out[0] = samples;
    out[1] = labels;
    
    return out;
}
