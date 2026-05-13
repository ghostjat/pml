<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Lib\SafeTensorsIO;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Radius Neighbors Classifier.
 * Classifies a sample based on a majority vote of all neighbors within a given physical radius, 
 * rather than a strict count of K neighbors.
 * * JIT & Memory Optimized:
 * - Uses AVX2 Boolean Masking (`less()`) to extract points within the radius natively in C.
 */
final class RadiusNeighborsClassifier implements Learner, Persistable
{
    private float $radius;
    private int $numClasses = 0;
    private ?Tensor $fitSamples = null;
    private ?Tensor $fitLabels  = null;

    public function __construct(float $radius = 1.0)
    {
        $this->radius = $radius;
    }

    public function train(Dataset $dataset): void
    {
        $this->fitSamples  = $dataset->samples();
        $this->fitLabels   = $dataset->labels();
        $this->numClasses  = (int)($this->fitLabels->max() + 1);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("RadiusNeighbors is not trained.");

        $testX   = $dataset->samples();
        $nTest   = $testX->shape()[0];
        $radiusSq = $this->radius * $this->radius;

        // [nTest, nTrain] pairwise squared distances — one BLAS call
        $distMat  = Tensor::pairwiseSqL2($testX, $this->fitSamples);
        $inRadius = $distMat->lessScalar($radiusSq);              // [nTest, nTrain] bool

        // [nTrain, K] one-hot labels → matmul → [nTest, K] vote counts (one GEMM call)
        $labelOneHot = Tensor::onehot($this->fitLabels, $this->numClasses);
        $voteCounts  = $inRadius->matmul($labelOneHot);
        $preds       = $voteCounts->argmaxAxis(1);                // [nTest]

        // Outlier mask: no in-radius neighbors → predict 0
        $noNeighbor = $inRadius->sumAxis(1)->lessScalar(1.0);
        $zeroVec    = Tensor::zeros($nTest);
        return $noNeighbor->where($zeroVec, $preds);
    }

    public function trained(): bool
    {
        return $this->fitSamples !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['radius' => $this->radius, 'numClasses' => $this->numClasses]));
        if ($this->fitSamples !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['fit_samples' => $this->fitSamples, 'fit_labels' => $this->fitLabels]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((float)$c['radius']);
        $i->numClasses = (int)($c['numClasses'] ?? 2);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->fitSamples = $t['fit_samples'] ?? null; $i->fitLabels = $t['fit_labels'] ?? null; }
        return $i;
    }
}
