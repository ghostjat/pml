## Purpose

This file contains the implementation of the Radial Basis Function (RBF) / Gaussian Kernel class for use in a machine learning framework. The RBF kernel is commonly used in Support Vector Machines (SVMs) and other kernel-based models to transform input data into a higher-dimensional space where it becomes easier to find hyperplanes that separate classes.

## Key Components

### Classes
- **RBF**: Implements the RBF / Gaussian Kernel.

### Functions / Methods
- **constructor(float $gamma = 1e-3)**: Initializes the RBF kernel with a gamma value.
  - Parameters:
    - `float $gamma`: The kernel coefficient. Defines how far the influence of a single training example reaches.
  
- **compute(Tensor $a, Tensor $b): Tensor**: Computes the RBF kernel for two sets of data points.
  - Parameters:
    - `Tensor $a`: Input tensor of shape `[N, d]` representing N data points with d features.
    - `Tensor $b`: Input tensor of shape `[M, d]` representing M data points with d features.
  - Returns: A tensor of shape `[N, M]` containing the RBF kernel values.

### Important Variables / Constants
- **$gamma**: The kernel coefficient that controls the influence distance of a single training example. Default value is `1e-3`.

## Inputs / Outputs

### For ML Components
- **Input Shapes**:
  - `Tensor $a`: Shape `[N, d]`
  - `Tensor $b`: Shape `[M, d]`

- **Output Format**: A tensor of shape `[N, M]` containing the RBF kernel values.

## Dependencies
- The class depends on the `Pml\Tensor` class for handling tensors and performing operations like squaring, summing, matrix multiplication, and exponentiation.

## Usage Notes

### How This File Integrates with the Rest of the Framework
This file should be used in conjunction with other components of the machine learning framework that utilize kernel methods, such as SVMs. The RBF kernel can be instantiated with a specific gamma value and then passed to these models for computing similarity matrices.

### Edge Cases
- If `gamma` is set too low, it may result in a very high influence range, potentially leading to overfitting.
- If `gamma` is set too high, it may result in a very narrow influence range, potentially leading to underfitting.

### Performance Considerations
- The implementation uses AVX2 instructions for efficient computation of the pairwise distances and exponential function. This makes it highly memory and performance optimized compared to using single PHP loops.
- The use of JIT (Just-In-Time) compilation can further enhance performance by optimizing code at runtime based on specific input data characteristics.

Overall, this RBF kernel class is a crucial component for enabling advanced machine learning models within the framework, providing efficient and effective kernel computations.