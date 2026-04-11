<?php
declare(strict_types=1);

namespace Pml\Datasets\Generators;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Agglomerate — generates a labeled dataset from a weighted mixture of child generators.
 * Each child contributes proportionally to its weight; labels are the child's index.
 *
 * JIT & Memory Optimized:
 * - All sample generation delegates to C-level Tensor ops inside each child generator.
 * - Final concatenation is a single C call (tensor_concat).
 */
final class Agglomerate implements Generator
{
    /** @var Generator[] */
    private array $generators;
    /** @var float[] normalized probabilities */
    private array $weights;

    /**
     * @param Generator[] $generators  Child generators keyed by class label
     * @param float[]     $weights     Weight per generator (will be normalized)
     */
    public function __construct(array $generators, array $weights = [])
    {
        if (empty($generators)) {
            throw new \InvalidArgumentException("Agglomerate requires at least one generator.");
        }

        if (empty($weights)) {
            $weights = array_fill(0, count($generators), 1.0);
        }

        if (count($generators) !== count($weights)) {
            throw new \InvalidArgumentException("Generators and weights counts must match.");
        }

        $total = array_sum($weights);
        $this->weights    = array_map(fn($w) => $w / $total, array_values($weights));
        $this->generators = array_values($generators);
    }

    public function generate(int $n): Dataset
    {
        $sampleParts = [];
        $labelParts  = [];
        $remaining   = $n;

        foreach ($this->generators as $i => $gen) {
            $count = ($i === count($this->generators) - 1)
                ? $remaining
                : (int) round($this->weights[$i] * $n);
            $count = max(1, min($count, $remaining));

            $sub    = $gen->generate($count);
            $sampleParts[] = $sub->samples();
            // Override labels with the generator index
            $labelParts[]  = Tensor::zeros($count)->addScalarInplace((float) $i);
            $remaining -= $count;

            if ($remaining <= 0) break;
        }

        return new Dataset(
            Tensor::concat($sampleParts, 0),
            Tensor::concat($labelParts, 0)
        );
    }
}
