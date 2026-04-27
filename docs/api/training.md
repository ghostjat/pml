# Training API

This section documents the training orchestration used for neural networks and general training loops.

## `Pml\Training\Trainer`

`Trainer` is a high-level training driver used for `TorchBackend` and `Sequential` models.

### Signature

```php
public function __construct(MLBackend $backend, TrainingArguments $args, ?LoggerInterface $logger = null)
public function addCallback(TrainerCallback $callback): void
public function train(Dataset $dataset, ?Dataset $validation = null): TrainingResult
```

### What it does

- Creates a `DataLoader` for mini-batching.
- Optionally shuffles the dataset.
- Runs epoch/batch loops for deep learning backends.
- Computes train and validation loss.
- Applies LR scheduling and checkpointing.
- Fires callback hooks at lifecycle events.

## `Pml\Training\TrainingArguments`

A typed container for training hyperparameters.

### Constructor

```php
public function __construct(
    int $epochs = 10,
    int $batchSize = 32,
    int $patience = 0,
    float $minDelta = 1e-4,
    float $learningRate = 0.001,
    string $lrSchedule = 'none',
    float $lrDecay = 0.1,
    int $lrStepSize = 5,
    int $warmupEpochs = 0,
    bool $mixedPrecision = false,
    ?string $outputDir = null,
    int $saveEvery = 0,
    bool $saveBest = true,
    int $logEvery = 1
)
```

### Methods

```php
public function toTrainOptions(?Dataset $validation = null): array
```

Returns a named options array suitable for `Sequential::train()`.

## Data loading

### `Pml\Data\DataLoader`

```php
public function __construct(Dataset $dataset, int $batchSize = 32, bool $shuffle = false, bool $dropLast = false, ?DataCollator $collator = null)
public function batches(): \Generator
public function steps(): int
public function batchSize(): int
public function dataset(): Dataset
```

`DataLoader::batches()` yields zero-copy `DataBatch` objects.

### `Pml\Data\DataBatch`

```php
public function __construct(Tensor $inputs, ?Tensor $labels = null, array $meta = [])
public function inputs(): Tensor
public function labels(): ?Tensor
public function meta(): array
public function hasLabels(): bool
public function size(): int
```

## Callbacks

### `Pml\Training\TrainerCallback`

Implement this interface to receive hooks during training.

```php
public function onTrainBegin(TrainingArguments $args, int $steps): void
public function onEpochBegin(int $epoch, int $epochs): void
public function onBatchEnd(int $step, float $batchLoss): void
public function onEpochEnd(int $epoch, float $trainLoss, ?float $valLoss): void
public function onTrainEnd(TrainingResult $result): void
```

## `Pml\Training\TrainingResult`

Stores summary information after training.

### Fields

- `epochsRun`
- `trainLossHistory`
- `valLossHistory`
- `bestValLoss`
- `earlyStopped`
- `elapsedSeconds`

## Example Usage

```php
use Pml\Training\Trainer;
use Pml\Training\TrainingArguments;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\Losses\MeanSquaredError;
use Pml\NeuralNetwork\Optimizers\Adam;

$dataset = Dataset::fromArray(
    [[0.0], [1.0], [2.0]], [0.0, 2.0, 4.0]
);

$model = new Sequential([
    new Dense(1, 10),
    new Dense(10, 1),
], new MeanSquaredError(), new Adam(0.01));

$args = new TrainingArguments(epochs: 20, batchSize: 1, patience: 5);
$trainer = new Trainer($backend, $args);
$result = $trainer->train($dataset);
```

## Performance notes

- `DataLoader` shuffles in-place using `Dataset::randomize()`.
- The trainer uses zero-copy batch views to avoid repeated copies.
- Checkpointing and validation can be enabled without changing the model code.

## Common mistakes

- Passing an unlabeled dataset to a learner that requires labels.
- Trying to use `DataLoader` on a dataset before `materialize()`.
- Ignoring the callback hooks if you need metrics or custom logging.
