# Sigmoidal.php

## Purpose
The `Sigmoidal.php` file contains a class that implements the Sigmoidal (Hyperbolic Tangent) kernel for use in Support Vector Machines (SVM). This kernel is designed to measure similarity between samples in high-dimensional spaces and is particularly useful when the data is non-linearly separable.

## Key Components
### Classes, Functions, Methods with Signatures
- **Class: Sigmoidal**
  - Implements the `Kernel` interface.
  - Contains a constructor for initializing the kernel parameters.
  - Contains a method `compute` that calculates the kernel value between two tensors.

- **Method: `__construct`**
  ```php
  public function __construct(
      private readonly float $gamma = 0.001,
      private readonly float $coef0 = 0.0
  ) {}
  ```
  - Initializes the `gamma` and `coef0` parameters, which control the shape of the kernel.

- **Method: `compute`**
  ```php
  public function compute(Tensor $a, Tensor $b): Tensor
  ```
  - Computes the Sigmoidal kernel between two tensors.
  - Parameters:
    - `$a`: A `Tensor` representing one set of samples.
    - `$b`: A `Tensor` representing another set of samples.
  - Returns: A `Tensor` containing the computed kernel values.

### Important Variables and Constants
- **Variables:**
  - `$gamma`: Controls the width of the Gaussian bell curve in the feature space. Default value is `0.001`.
  - `$coef0`: Bias term added to the result. Default value is `0.0`.

## Inputs / Outputs
### For ML Components
- **Inputs:**
  - Two tensors `$a` and `$b`, each of shape `(n_samples, n_features)`. These tensors represent the samples for which the kernel values are to be computed.

- **Outputs:**
  - A tensor of shape `(n_samples_a, n_samples_b)` containing the Sigmoidal kernel values between each pair of samples from `$a` and `$b`.

### For Utility Files
- **Parameters:** None.
- **Return Values:** None (methods return tensors).

## Dependencies
This class depends on the following components:
- `Pml\Tensor`: A tensor library for handling multi-dimensional arrays efficiently.

## Usage Notes
### How This File Integrates with the Rest of the Framework
The `Sigmoidal` kernel can be used in SVM models to transform non-linearly separable data into a higher-dimensional space where it becomes linearly separable. It is typically used when the relationship between the input features and the target variable cannot be easily captured by linear functions.

### Edge Cases, Performance Considerations
- **Edge Cases:**
  - If `gamma` is set to zero, the kernel will degenerate into a simple dot product, which may not capture complex relationships in the data.
  - Large values of `coef0` can lead to a more biased fit, potentially overfitting the training data.

- **Performance Considerations:**
  - The computation involves matrix multiplication and element-wise operations. For large datasets, these operations can be computationally expensive. Optimizations such as parallel processing or using specialized libraries (e.g., CUDA for GPU acceleration) may be necessary to handle large-scale data efficiently.

### Example Usage
Here's an example of how you might use the `Sigmoidal` kernel in a machine learning model:

```php
use Pml\Kernels\SVM\Sigmoidal;
use Pml\Tensor;

// Initialize tensors
$a = new Tensor([[1, 2], [3, 4]]);
$b = new Tensor([[5, 6], [7, 8]]);

// Create a Sigmoidal kernel instance
$kernel = new Sigmoidal(gamma: 0.1, coef0: 0.5);

// Compute the kernel values
$result = $kernel->compute($a, $b);
```

This example initializes two tensors and computes their Sigmoidal kernel with custom `gamma` and `coef0` parameters.