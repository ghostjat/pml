# DataType.php

## Purpose

The `DataType` class in the Pml (PHP Machine Learning) framework is designed to describe and categorize the data types of columns within a dataset. This helps in ensuring that data is handled appropriately during various stages of machine learning tasks, from preprocessing to model training.

## Key Components

### Classes, Functions, Methods with Signatures

- **Class: `DataType`**
  - Represents a data type descriptor for dataset columns.
  - JIT optimized for performance by resolving constants at compile time and avoiding heap allocations.

#### Public Constants
- `CONTINUOUS`: Indicates that the data is continuous (float64 numeric).
- `CATEGORICAL`: Indicates that the data is categorical (string label).
- `IMAGE`: Indicates that the data is an image (pixel tensor).
- `OTHER`: Indicates that the data is of an unknown or opaque type.

#### Private Constructor
- `private function __construct(private readonly int $code)`
  - Initializes a new instance with the given code.
  - Ensures that instances of this class cannot be created outside of static factory methods.

#### Public Static Methods
- `public static function continuous(): self`
  - Returns an instance representing a continuous data type.
- `public static function categorical(): self`
  - Returns an instance representing a categorical data type.
- `public static function image(): self`
  - Returns an instance representing an image data type.
- `public static function other(): self`
  - Returns an instance representing any other (opaque) data type.

#### Public Methods
- `public function isContinuous(): bool`
  - Checks if the instance represents a continuous data type.
- `public function isCategorical(): bool`
  - Checks if the instance represents a categorical data type.
- `public function isImage(): bool`
  - Checks if the instance represents an image data type.
- `public function code(): int`
  - Returns the numeric code of the data type.
- `public function __toString(): string`
  - Provides a human-readable string representation of the data type.

### Important Variables and Constants

- **Constants**: `CONTINUOUS`, `CATEGORICAL`, `IMAGE`, `OTHER`
- **Properties**: `$code`
  - Holds the numeric code representing the data type.

## Inputs / Outputs

### For ML Components

- **Expected Input Shapes**: The input should be a column of data from a dataset, represented as an array or similar structure.
- **Data Types**: The data can be of various types such as float64 (continuous), string (categorical), pixel tensors (image), or opaque blobs (other).
- **Output Formats**: The output will be an instance of the `DataType` class representing the type of the input column.

### For Utility Files

- **Parameters**:
  - None.
- **Return Values**:
  - An instance of the `DataType` class corresponding to the specified data type.

## Dependencies

- **No external dependencies**: All functionality is self-contained within this class and does not rely on any external libraries or packages.

## Usage Notes

### How This File Integrates with the Rest of the Framework

The `DataType` class plays a critical role in the preprocessing stage of machine learning workflows. It allows for easy categorization and handling of different data types within datasets, which is essential for tasks such as feature engineering, model training, and evaluation.

### Edge Cases

- **Unknown Data Type**: If an unknown data type is encountered during runtime, it should be handled appropriately by defaulting to the `OTHER` data type or raising a custom exception.
- **Performance Considerations**: The JIT optimization ensures that the class methods are compiled and optimized at compile time, minimizing heap allocations and improving performance.

### Example Usage

```php
use Pml\DataType;

$column1 = [1.2, 3.4, 5.6]; // Continuous data
$column2 = ['apple', 'banana', 'cherry']; // Categorical data
$column3 = [[0, 0, 0], [255, 0, 0]]; // Image data

$type1 = DataType::continuous();
$type2 = DataType::categorical();
$type3 = DataType::image();

echo $type1; // Output: continuous
echo $type2; // Output: categorical
echo $type3; // Output: image
```

This example demonstrates how to create instances of the `DataType` class and check their types.