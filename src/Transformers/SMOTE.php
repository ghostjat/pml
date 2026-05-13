<?php

declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Synthetic Minority Over-sampling Technique (SMOTE).
 * Generates synthetic samples for the minority class by interpolating between nearest neighbors.
 * * JIT & Memory Optimized:
 * - Employs OpenBLAS to calculate distance matrices for the minority class instantly.
 * - Generates synthetic interpolations in massive C-memory batches using vectorized arithmetic.
 */
final class SMOTE implements Transformer
{
    private int $k;
    private int $amount;

    /**
     * @param int $k The number of nearest neighbors to use for interpolation.
     * @param int $amount The number of synthetic samples to generate.
     */
    public function __construct(int $amount = 100, int $k = 5)
    {
        $this->amount = $amount;
        $this->k = $k;
    }

    public function fit(Dataset $dataset): void
    {
        // Stateless Transformer
    }

    public function transform(Dataset $dataset): Dataset
    {
        $x = $dataset->samples();
        $y = $dataset->labels();

        if ($y === null) {
            throw new \InvalidArgumentException("SMOTE requires labeled data.");
        }

        // 1. Find the minority class
        $counts = $y->bincount()->toFlatArray();
        $minorityClass = array_keys($counts, min(array_filter($counts)))[0];

        // 2. Extract minority samples natively in C
        $minorityMask = $y->equal(Tensor::zeros(1)->addScalarInplace($minorityClass));
        $xMin = $x->booleanIndex($minorityMask);
        $nMin = $xMin->shape()[0];

        if ($nMin <= $this->k) {
            throw new RuntimeException("Not enough minority samples to use K={$this->k}.");
        }

        // 3. Vectorized pairwise distances for minority class: D = A^2 + B^2 - 2AB
        $xSq = $xMin->square()->sumAxis(1)->expandDims(1);
        $xSqT = $xSq->transpose();
        $distSq = $xMin->matmul($xMin->transpose())
                       ->mulScalarInplace(-2.0)
                       ->addInplace($xSq)
                       ->addInplace($xSqT)
                       ->clip(0.0, INF);

        // Set self-distance to INF to avoid picking self as neighbor
        $infDiag = Tensor::eye($nMin)->mulScalarInplace(INF);
        $distSq->addInplace($infDiag);

        // 4. Generate Synthetic Samples purely in Vectorized C-Math
        $synthetic = [];
        for ($i = 0; $i < $this->amount; $i++) {
            // Pick a random minority point
            $idxA = mt_rand(0, $nMin - 1);
            $pointA = $xMin->row($idxA);

            // Find its K-nearest neighbors and pick one randomly
            $sortedIndices = $distSq->row($idxA)->argsort()->slice(0, 0, $this->k)->toFlatArray();
            $idxB = $sortedIndices[mt_rand(0, $this->k - 1)];
            $pointB = $xMin->row($idxB);

            // Interpolate: New = A + rand(0, 1) * (B - A)
            $step = mt_rand() / mt_getrandmax();
            $diff = $pointB->sub($pointA)->mulScalarInplace($step);
            $synthetic[] = $pointA->addInplace($diff);
        }

        // 5. Append generated data
        $xSyn = Tensor::concat($synthetic, 0);
        $ySyn = Tensor::zeros($this->amount)->addScalarInplace($minorityClass);

        $xNew = Tensor::concat([$x, $xSyn], 0);
        $yNew = Tensor::concat([$y, $ySyn], 0);

        return new Dataset($xNew, $yNew);
    }

    public function fitted(): bool
    {
        return true;
    }
}