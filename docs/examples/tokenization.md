# Tokenization Example

Use the `Pml\Inference\Tokenizer` wrapper to load a HuggingFace tokenizer and encode text.

```php
<?php
require 'vendor/autoload.php';

use Pml\Inference\Tokenizer;

$tokenizer = Tokenizer::fromJson('models/gpt2/tokenizer.json');

$ids = $tokenizer->encode('Hello world', addBos: true);
printf("IDs: %s\n", json_encode($ids));

$text = $tokenizer->decode($ids);
printf("Decoded: %s\n", $text);

$batch = $tokenizer->encodeBatch(['Hello', 'World'], addBos: true, maxLen: 10);
printf("Batch shape: %s\n", json_encode($batch->shape()));
```

## Notes

- `encodeBatch()` returns a tensor padded to `maxLen`.
- The tokenizer supports GPT-2 style JSON and legacy `vocab.json` + `merges.txt` formats.
- Use `bosId()`, `eosId()`, and `padId()` to inspect special tokens.
