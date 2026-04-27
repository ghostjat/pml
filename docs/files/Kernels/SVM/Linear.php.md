# File: Linear.php

## Purpose

The `Linear.php` file is part of the PML (Palm Machine Learning) framework and contains a class for computing the linear kernel between two matrices. The linear kernel is a fundamental component in many machine learning algorithms, such as Support Vector Machines (SVMs), where it measures the similarity between two vectors.

## Key Components

### Classes
- **Linear**: A final class implementing the `Kernel` interface that computes the dot product (linear kernel) between two matrices using optimized matrix multiplication through OpenBLAS.

### Functions / Methods
- **compute(Tensor $a, Tensor $b): Tensor**
  - Computes the linear kernel between two tensors `a` and `b`.
  - **Parameters:**
    - `$a`: The first tensor of type `Tensor`.
    - `$b`: The second tensor of type `Tensor`.
  - **Returns:** A new `Tensor` representing the dot product of matrices `a` and `b`.

### Important Variables / Constants
- None.

## Inputs / Outputs

### For ML Components
- **Inputs:**
  - Two tensors (`$a` and `$b`) of compatible shapes for matrix multiplication.
    - Example input shape for both tensors might be `[n_samples, n_features]`.
- **Outputs:**
  - A new `Tensor` representing the dot product (linear kernel) of the two input tensors. The output tensor will have a shape depending on the dimensions of the inputs.

### For Utility Files
- None.

## Dependencies

- **Pml\Tensor**: The `Linear` class uses this class for handling tensor operations.
- **OpenBLAS**: Used for efficient matrix multiplication via the `matmul` method in the `Tensor` class.

## Usage Notes

### Integration with the Framework
The `Linear` class is designed to be used within the PML framework, particularly in SVM implementations where kernel computations are essential. It relies on the `Pml\Tensor` class for tensor operations and optimization.

### Edge Cases
- Ensure that both input tensors (`$a` and `$b`) have compatible shapes for matrix multiplication.
  - If the number of features in `$a` does not match the number of samples (or vice versa) in `$b`, an error will occur during execution.
  
### Performance Considerations
- The use of OpenBLAS ensures that the matrix multiplication is optimized for memory usage and execution speed, making this kernel suitable for large datasets.
- Ensure that the tensors are pre-allocated with the correct data types to avoid runtime errors.

## Example Usage

Here's an example of how you might use the `Linear` class in a hypothetical SVM implementation:

```php
use Pml\Kernels\SVM\Linear;
use Pml\Tensor;

// Create two sample matrices
$a = new Tensor([[1, 2], [3, 4]]);
$b = new Tensor([[5, 6], [7, 8]]);

// Create a Linear kernel instance
$kernel = new Linear();

// Compute the linear kernel between the two matrices
$result = $kernel->compute($a, $b);

echo $result; // Output should be the dot product of a and b
```

This example demonstrates how to compute the linear kernel between two sample matrices using the `Linear` class. The resulting tensor will contain the computed dot products, which can then be used in further steps such as training an SVM model.