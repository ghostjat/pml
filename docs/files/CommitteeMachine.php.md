## Purpose

The `CommitteeMachine.php` file defines the `CommitteeMachine` class, which is a component of an ML framework. This class represents a weighted ensemble of heterogeneous estimators (models). It aggregates predictions from each estimator by taking a weighted average for probabilistic models and a weighted hard vote for non-probabilistic classifiers.

The Committee Machine is designed to be JIT (Just-In-Time) and memory-optimized, using PHP scalars for weights and leveraging C-level operations for accumulation. Each estimator trains independently, ensuring zero shared state between them.

## Key Components

### Classes, Functions, Methods with Signatures
- **CommitteeMachine**: Main class implementing the `Learner` interface.
  - `__construct(array $members)`: Constructor to initialize the committee machine with a set of estimators and their corresponding weights.
  - `train(Dataset $dataset): void`: Method to train each estimator in the ensemble using the provided dataset.
  - `predict(Dataset $dataset): Tensor`: Method to make predictions by aggregating predictions from all estimators, weighted accordingly.
  - `trained(): bool`: Getter method to check if the committee machine has been trained.

### Important Variables and Constants
- **$members**: An array of associative arrays where each element contains an estimator (implementing `Learner`) and its weight. This is initialized in the constructor.
- **$trained**: A boolean flag indicating whether the committee machine has been trained.

## Inputs / Outputs

### For ML Components:
- **Input**:
  - `Dataset $dataset`: The dataset used for training or making predictions.
- **Output**:
  - `predict(Dataset $dataset): Tensor`: Returns a tensor of predictions, aggregated based on the weights of each estimator.

### For Utility Files:
- **Parameters**: None directly related to parameters, as this is an ML component.
- **Return Values**: 
  - `train()`: No return value (void).
  - `predict()`: A tensor representing the aggregated predictions.
  - `trained()`: A boolean indicating whether the machine has been trained.

## Dependencies

### Internal Dependencies
- Implements the `Learner` interface from the same namespace (`Pml\Interfaces\Learner`).

### External Dependencies
- Uses the `Dataset` class and `Tensor` class, which are part of the ML framework's utility classes.

## Usage Notes

### Integration with the Rest of the Framework
- **Training**: The `CommitteeMachine` should be instantiated with a list of estimators and their weights. It then trains each estimator independently using the provided dataset.
- **Prediction**: Once trained, predictions can be made by calling the `predict()` method. The predictions are aggregated based on the weights assigned to each estimator.

### Edge Cases
- If no members are provided in the constructor, an `InvalidArgumentException` is thrown.
- Attempting to predict without training results in a `RuntimeException`.

### Performance Considerations
- **Memory Optimization**: Using PHP scalars for weights and C-level operations for accumulation reduces memory overhead and improves performance.
- **Independent Training**: Ensuring each estimator trains independently prevents shared state, which can be crucial for scalability and independence.

This detailed documentation covers the purpose, key components, inputs/outputs, dependencies, and usage notes of the `CommitteeMachine.php` file within an ML framework.