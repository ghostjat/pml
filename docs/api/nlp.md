# NLP API

This section documents tokenization and text preprocessing.

## Tokenizer interface

### `Pml\Tokenizers\Tokenizer`

```php
public function tokenize(string $text): array;
```

## Built-in tokenizers

- `Pml\Tokenizers\Whitespace`
- `Pml\Tokenizers\Word`
- `Pml\Tokenizers\WordStemmer`
- `Pml\Tokenizers\Sentence`
- `Pml\Tokenizers\NGram`
- `Pml\Tokenizers\KSkipNGram`
- `Pml\Tokenizers\CharGram`

Each tokenizer returns a string token array.

## Text transformers

### `Pml\Transformers\WordCountVectorizer`

A BoW transformer that builds a vocabulary in C and produces dense count features.

```php
public function __construct(?int $maxFeatures = null, string $textColumn = 'text')
public function fit(Dataset $data): void
public function transform(Dataset $data): Dataset
public function fitTransform(Dataset $data): Dataset
public function fitted(): bool
public function vocabSize(): int
public function vocabPtr(): ?\FFI\CData
```

`WordCountVectorizer` requires ETL mode and a text column.

### `Pml\Transformers\TfIdfTransformer`

```php
public function fit(Dataset $dataset): void
public function transform(Dataset $dataset): Dataset
public function fitted(): bool
public function getStateDict(string $prefix = ''): array
public function loadStateDict(array $dict, string $prefix = ''): void
```

This transformer computes TF-IDF in C using tensor operations.

### `Pml\Transformers\TextNormalizer`

```php
public function fit(Dataset $dataset): void
public function transform(Dataset $dataset): Dataset
public static function normalize(string $text): string
public function fitted(): bool
```

`TextNormalizer` normalizes text by lowercasing, stripping punctuation, and collapsing whitespace.

## Dataset convenience

`Dataset::bagOfWords($column, ?int $maxFeatures = null)` is a helper that fits and transforms a `WordCountVectorizer` on the given text column.

## Example usage

```php
use Pml\Dataset;
use Pml\Transformers\WordCountVectorizer;

$ds = Dataset::load('datasets/emails.csv', true)
    ->withLabelColumn(0);

$vectorizer = new WordCountVectorizer(maxFeatures: 1000, textColumn: 'body');
$dataset = $vectorizer->fitTransform($ds);

print_r($dataset->samples()->shape());
```

## When to use it

- Use `WordCountVectorizer` for bag-of-words features from text columns.
- Use `TfIdfTransformer` after `WordCountVectorizer` to reweight term frequencies.
- Use `TextNormalizer` before tokenization to standardize raw strings.

## Common mistakes

- Trying to fit `WordCountVectorizer` on a tensor-mode dataset.
- Assuming `Dataset::bagOfWords()` modifies the original dataset in-place; it returns a new dataset.
- Forgetting to set the label column before `extractLabelTensor()` or `materialize()`.
