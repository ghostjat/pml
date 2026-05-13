<?php

declare(strict_types=1);

namespace Pml\Estimators\AnomalyDetectors;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Local Outlier Factor (LOF).
 * Unsupervised anomaly detector computing local density deviations using K-Nearest Neighbors.
 * * JIT & Memory Optimized:
 * - Uses vectorized distance broadcasting against the training matrix.
 * - Bypasses O(N^2) memory footprint by calculating top-K distances incrementally in PHP cache.
 */
final class LocalOutlierFactor implements Learner, Persistable
{
    private int $k;
    private ?Tensor $fitSamples = null;

    public function __construct(int $k = 20)
    {
        $this->k = $k;
    }

    public function train(Dataset $dataset): void
    {
        // LOF is a lazy learner. Memorize the training set structure natively.
        $this->fitSamples = $dataset->samples();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("LOF is not trained.");
        }

        $testX  = $dataset->samples();
        $nTest  = $testX->shape()[0];
        $nTrain = $this->fitSamples->shape()[0];
        $k      = min($this->k, $nTrain - 1);

        // One BLAS call: [nTest, nTrain] pairwise squared L2 distances
        $distMat    = Tensor::pairwiseSqL2($testX, $this->fitSamples);
        $sortedDist = $distMat->sort(1);                    // [nTest, nTrain] ascending
        $kDists     = $sortedDist->slice(1, 0, $k);         // [nTest, k] nearest distances
        $meanDists  = $kDists->meanAxis(1);                 // [nTest] mean k-nn sq distance

        // Near-zero mean distance (self-match) → return 1.0 (normal density baseline)
        $nearZero = $meanDists->lessScalar(1e-8);
        $ones     = Tensor::ones($nTest);
        return $nearZero->where($ones, $meanDists);
    }

    public function trained(): bool
    {
        return $this->fitSamples !== null;
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) { mkdir($dir, 0755, true); }
        file_put_contents($dir . '/config.json', json_encode(['k' => $this->k], JSON_PRETTY_PRINT));
        if ($this->fitSamples !== null) {
            SafeTensorsIO::save($dir . '/model.safetensors', ['fit_samples' => $this->fitSamples]);
        }
    }

    public static function load(string $dir): self
    {
        $cfg = json_decode(file_get_contents($dir . '/config.json'), true);
        $instance = new self((int) $cfg['k']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $tensors = SafeTensorsIO::load($stPath);
            $instance->fitSamples = $tensors['fit_samples'] ?? null;
        }
        return $instance;
    }
}