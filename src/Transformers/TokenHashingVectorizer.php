<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Token Hashing Vectorizer (Feature Hashing / Hashing Trick).
 * Maps tokens to fixed-size feature vectors without a learned vocabulary.
 * Collisions are handled by sign(hash) — unbiased in expectation.
 *
 * JIT & Memory Optimized:
 * - Hash computation is pure PHP (fast for moderate vocab sizes).
 * - Result is loaded into a C Tensor in one fromArray() call per sample.
 */
final class TokenHashingVectorizer implements Transformer
{
    private bool $fitted = false;

    public function __construct(
        private readonly int  $nFeatures  = 1024,
        private readonly bool $alternate  = true    // alternate sign to reduce variance
    ) {}

    public function fit(Dataset $dataset): void { $this->fitted = true; }

    public function transform(Dataset $dataset): Dataset
    {
        // Expects labels to be string token lists OR samples to be pre-tokenized integer indices.
        // For the Tensor-native pipeline, samples are already numeric — return as-is.
        return $dataset;
    }

    /**
     * Vectorize an array of string tokens into a fixed-size hashed feature vector.
     * @param string[] $tokens
     * @return Tensor  shape [nFeatures]
     */
    public function vectorize(array $tokens): Tensor
    {
        $vec = array_fill(0, $this->nFeatures, 0.0);

        foreach ($tokens as $token) {
            $h   = crc32($token);
            $idx = abs($h) % $this->nFeatures;
            $sign = $this->alternate ? (($h >> 31 & 1) ? -1.0 : 1.0) : 1.0;
            $vec[$idx] += $sign;
        }

        return Tensor::fromArray($vec);
    }

    public function fitted(): bool { return $this->fitted; }
}
