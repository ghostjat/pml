<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * One-Hot Label Encoder.
 * Converts integer class labels into a sparse binary matrix for Categorical Cross Entropy.
 * * JIT & Memory Optimized:
 * - Employs a massive C-level Broadcasting trick using `tensor_equal`.
 * - Transforms millions of labels instantly with zero PHP iteration loops.
 */
final class OneHotLabelEncoder implements Transformer
{
    private ?Tensor $categories = null;

    /**
     * Plain-PHP field that survives Pipeline::save()'s stripTensors() pass.
     * Stores [cat0, cat1, ...] as float[] so __serialize/__unserialize can
     * reconstruct $categories without needing the live Tensor.
     * @var float[]
     */
    private array $categoryValues = [];

    public function fit(Dataset $dataset): void
    {
        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("Dataset must be labeled to fit the OneHotLabelEncoder.");
        }

        // 1. Extract unique categories and sort them: Output is 1D Tensor [K]
        $unique = $y->unique()->sort(0);

        // 2. Cache as plain PHP array (survives stripTensors Tensor-nulling)
        $this->categoryValues = array_map('floatval', $unique->toFlatArray());

        // 3. Expand to [1, K] to prepare for Hardware Broadcasting
        $this->categories = $unique->expandDims(0);
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (!$this->fitted()) {
            throw new \RuntimeException("Encoder has not been fitted.");
        }

        $y = $dataset->labels();
        if ($y === null) {
            return $dataset; // Nothing to encode
        }

        // THE BROADCASTING TRICK:
        // 1. Expand the [N] labels to [N, 1]
        $yExpanded = $y->expandDims(1);

        // 2. Check equality against the [1, K] categories tensor.
        // The C engine automatically broadcasts both tensors to [N, K] and evaluates equality.
        // This generates a 100% perfect One-Hot matrix instantly without copying memory into PHP!
        $oneHotLabels = $yExpanded->equal($this->categories);

        return new Dataset($dataset->samples(), $oneHotLabels);
    }

    public function fitted(): bool
    {
        return $this->categories !== null;
    }
    
    /**
     * Returns the number of distinct categories found during fitting.
     */
    public function numCategories(): int
    {
        return $this->fitted() ? $this->categories->size() : 0;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Serialization — persist category count so fitted() stays true after load.
    //
    // Pipeline::save() calls stripTensors() which nulls $categories (Tensor),
    // making fitted() return false on reload.  We instead store the category
    // values as a plain PHP float[] and reconstruct the Tensor on unserialize.
    // ─────────────────────────────────────────────────────────────────────────

    public function __serialize(): array
    {
        // $categoryValues is a plain float[] — always set by fit(), NOT nulled
        // by Pipeline::save()'s stripTensors() (which only strips Tensor props).
        return ['categoryValues' => $this->categoryValues];
    }

    public function __unserialize(array $data): void
    {
        $this->categoryValues = $data['categoryValues'] ?? [];
        if (!empty($this->categoryValues)) {
            $this->categories = Tensor::fromArray([$this->categoryValues]); // [1, K]
        }
    }
}