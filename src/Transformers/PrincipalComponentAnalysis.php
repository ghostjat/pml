<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Principal Component Analysis (PCA).
 * Reduces dataset dimensionality while preserving variance.
 * * JIT & Memory Optimized:
 * - Direct LAPACKE SVD extraction.
 * - Zero-Copy slicing for principal components.
 */
final class PrincipalComponentAnalysis implements Learner, Persistable
{
    private int $nComponents;
    private ?Tensor $components = null;
    private ?Tensor $means = null;

    public function __construct(int $nComponents)
    {
        if ($nComponents < 1) {
            throw new \InvalidArgumentException("Number of components must be >= 1.");
        }
        $this->nComponents = $nComponents;
    }

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $x = $dataset->samples();

        // 1. Calculate column means for centering (Shape: [D])
        $this->means = $x->meanAxis(0);

        // 2. Center the dataset (AVX2 Broadcasting)
        $centered = $x->sub($this->means);

        // 3. Compute Singular Value Decomposition (SVD)
        $svd = $centered->svd();

        // 4. The Principal Components are the top K rows of V^T
        // Slice operates in <0.01ms as a zero-copy pointer adjustment. We copy() to safely own it.
        $this->components = $svd['Vt']->slice(0, 0, $this->nComponents)->copy();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("PCA has not been fitted.");
        }

        // 1. Center the inference data
        $centered = $dataset->samples()->sub($this->means);

        // 2. Project onto the principal components: X_c * V
        // $this->components is V^T, so we transpose it back to V.
        return $centered->matmul($this->components->transpose());
    }

    public function trained(): bool
    {
        return $this->components !== null;
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) { mkdir($dir, 0755, true); }

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode(
                ['class' => self::class, 'nComponents' => $this->nComponents],
                \JSON_PRETTY_PRINT
            )
        );

        if ($this->components !== null) {
            SafeTensorsIO::save(
                $dir . \DIRECTORY_SEPARATOR . 'model.safetensors',
                ['components' => $this->components, 'means' => $this->means]
            );
        }
    }

    public static function load(string $dir): self
    {
        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("PrincipalComponentAnalysis::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self((int) $config['nComponents']);

        $stPath = $dir . \DIRECTORY_SEPARATOR . 'model.safetensors';
        if (is_file($stPath)) {
            $tensors = SafeTensorsIO::load($stPath);
            $instance->components = $tensors['components'] ?? null;
            $instance->means      = $tensors['means']      ?? null;
        }

        return $instance;
    }
}