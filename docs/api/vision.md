# Vision API

This section documents the image preprocessing utilities in the framework.

## `Pml\Transformers\ImageVectorizer`

A transformer that flattens image tensors and normalizes pixel values.

### Signature

```php
public function __construct(float $maxPixelValue = 255.0)
public function fit(Dataset $dataset): void
public function transform(Dataset $dataset): Dataset
public function fitted(): bool
```

### What it does

- Accepts a dataset whose samples are image tensors.
- If the dataset is already 2-D, it normalizes values by `maxPixelValue`.
- If the dataset is 3-D or 4-D, it flattens each sample and normalizes.

## Image transformer list

The repository also contains the following image-related transformers:

- `Pml\Transformers\ImageResizer`
- `Pml\Transformers\ImageRotator`
- `Pml\Transformers\ImageCropper`
- `Pml\Transformers\ImageVectorizer`
- `Pml\Transformers\ColorSpaceConverter`

These components are designed to work on tensorized image data before model training.

## Example usage

```php
use Pml\Dataset;
use Pml\Transformers\ImageVectorizer;

$dataset = Dataset::fromArray(
    [
        [[[0.0], [128.0]], [[255.0], [64.0]]],
        [[[16.0], [32.0]], [[48.0], [80.0]]],
    ],
    [0, 1]
);

$vectorizer = new ImageVectorizer(maxPixelValue: 255.0);
$vectorizer->fit($dataset);
$dataset = $vectorizer->transform($dataset);

print_r($dataset->samples()->shape());
```

## Common mistakes

- Passing a text dataset to image transformers.
- Assuming the transformer will convert dataset labels; image transformers only modify `samples`.
- Forgetting to call `fit()` before `transform()` when a transformer requires it.
