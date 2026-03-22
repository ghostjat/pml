<?php
declare(strict_types=1);

namespace Pml\Generation;

use Pml\Tensor;
use Pml\Ops;


// ═══════════════════════════════════════════════════════════════════════════
//  SIMPLE BYTE-PAIR TOKENIZER STUB
//  A real implementation would use a .model file (sentencepiece/tiktoken).
//  This stub demonstrates the interface the LLM expects.
// ═══════════════════════════════════════════════════════════════════════════

class SimpleTokenizer
{
    private array $vocab;       // token → id
    private array $invVocab;    // id → token

    public const BOS = 1;
    public const EOS = 2;
    public const UNK = 0;

    public function __construct(string $vocabJsonPath)
    {
        if (!file_exists($vocabJsonPath)) {
            throw new \RuntimeException("Vocab file not found: {$vocabJsonPath}");
        }
        $this->vocab    = json_decode(file_get_contents($vocabJsonPath), true, flags: JSON_THROW_ON_ERROR);
        $this->invVocab = array_flip($this->vocab);
    }

    /**
     * Encode text → token IDs. Wraps with BOS/EOS.
     * Real implementation: use tiktoken or sentencepiece bindings.
     */
    public function encode(string $text, bool $addBos = true, bool $addEos = false): array
    {
        $tokens = $addBos ? [self::BOS] : [];

        // Naive whitespace tokenizer — replace with BPE in production
        foreach (preg_split('/(\s+)/', $text, flags: PREG_SPLIT_DELIM_CAPTURE | PREG_SPLIT_NO_EMPTY) as $piece) {
            $tokens[] = $this->vocab[$piece] ?? self::UNK;
        }

        if ($addEos) $tokens[] = self::EOS;
        return $tokens;
    }

    public function decode(array $tokenIds): string
    {
        $pieces = [];
        foreach ($tokenIds as $id) {
            if ($id === self::BOS || $id === self::EOS) continue;
            $pieces[] = $this->invVocab[$id] ?? '';
        }
        return implode('', $pieces);
    }

    public function decodeToken(int $id): string
    {
        return $this->invVocab[$id] ?? '';
    }

    public function vocabSize(): int { return count($this->vocab); }
}
