<?php

declare(strict_types=1);

namespace Pml\Estimators\Clusterers;

use Pml\Interfaces\Learner;
use Pml\Lib\SafeTensorsIO;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;

/**
 * K-Means Clustering.
 * Groups data into K distinct clusters.
 * * JIT & Memory Optimized:
 * - 100% Vectorized Expectation-Maximization loop.
 * - Uses AVX2 Boolean Masking to calculate cluster averages without flattening or iterating.
 */
final class KMeans implements Learner, Persistable
{
    private int $k;
    private int $maxIter;
    private float $tolerance;
    private ?Tensor $centroids = null;

    public function __construct(int $k = 3, int $maxIter = 300, float $tolerance = 1e-4)
    {
        $this->k = $k;
        $this->maxIter = $maxIter;
        $this->tolerance = $tolerance;
    }

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $x = $dataset->samples();
        $n = $x->shape()[0];

        // 1. Initialize centroids: random uniform scores → argsort → take first K rows.
        // Equivalent to sampling K rows without replacement; avoids randomChoice which
        // is unimplemented in the current C engine.
        $shuffleIdx      = Tensor::randomUniform([$n], 0.0, 1.0)->argsort();
        $seedIdx         = $shuffleIdx->slice(0, 0, $this->k);
        $this->centroids = $x->take($seedIdx, 0);

        for ($iter = 0; $iter < $this->maxIter; $iter++) {
            // E-step: assign every point to its nearest centroid (one C call, AVX2 + OMP)
            $assignments = Tensor::kmeansAssign($x, $this->centroids);

            // M-step: recompute centroids from assignments (one C call, empty-cluster safe)
            $newCentroids = Tensor::kmeansCentroids($x, $assignments, $this->k, $this->centroids);

            // Convergence check: max centroid shift
            $shift = $this->centroids->sub($newCentroids)->abs()->max();
            $this->centroids = $newCentroids;

            if ($shift < $this->tolerance) {
                break;
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("K-Means has not been fitted.");
        }

        return Tensor::kmeansAssign($dataset->samples(), $this->centroids);
    }

    public function trained(): bool
    {
        return $this->centroids !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['k'=>$this->k,'maxIter'=>$this->maxIter,'tolerance'=>$this->tolerance]));
        if ($this->centroids !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['centroids' => $this->centroids]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int)$c['k'], (int)$c['maxIter'], (float)$c['tolerance']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->centroids = $t['centroids'] ?? null; }
        return $i;
    }
}
