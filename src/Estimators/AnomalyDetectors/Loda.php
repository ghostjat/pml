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
 * LODA (Lightweight On-line Detector of Anomalies).
 * Highly efficient density estimator using random 1D projections and histograms.
 * * JIT & Memory Optimized:
 * - Extremely lightweight C-Level execution. Uses OpenBLAS `matmul` to project the entire dataset
 * onto random hyperplanes instantly.
 */
final class Loda implements Learner, Persistable
{
    private int $nProjections;
    private int $bins;
    
    private ?Tensor $projections = null;
    private array $histograms = [];
    private array $binEdges = [];

    public function __construct(int $nProjections = 100, int $bins = 10)
    {
        $this->nProjections = $nProjections;
        $this->bins = $bins;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $features = $x->shape()[1];
        $n = (float) $x->shape()[0];

        // 1. Generate sparse random projection vectors: [Features, Projections]
        // Standard normal distributed.
        $this->projections = Tensor::randomNormal([$features, $this->nProjections], 0.0, 1.0);

        // 2. Project dataset onto vectors natively in C: [N, Features] * [Features, Projections] -> [N, Projections]
        $z = $x->matmul($this->projections);

        // 3. Build 1D Histograms for each projection vector
        for ($p = 0; $p < $this->nProjections; $p++) {
            $col = $z->col($p)->toFlatArray();
            
            $min = min($col);
            $max = max($col) + 1e-8; // Prevent OutOfBounds
            $binWidth = ($max - $min) / $this->bins;
            
            $counts = array_fill(0, $this->bins, 0);

            foreach ($col as $val) {
                $binIdx = (int) floor(($val - $min) / $binWidth);
                $counts[min($binIdx, $this->bins - 1)]++;
            }

            // Convert counts to Log-Probabilities
            $logProbs = [];
            foreach ($counts as $count) {
                // Add epsilon for zero-count bins to prevent log(0)
                $logProbs[] = -log(($count / $n) + 1e-8); 
            }

            $this->histograms[$p] = $logProbs;
            $this->binEdges[$p] = ['min' => $min, 'width' => $binWidth];
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("LODA is not trained.");
        }

        $x    = $dataset->samples();
        $z    = $x->matmul($this->projections);      // [N, P] — one BLAS call
        $rows = $z->shape()[0];
        $totalScore = Tensor::zeros($rows);

        // O(P) loop: each iteration is O(N) in C — no PHP per-sample work
        $sentinel = (float) $this->bins;   // index of OOB sentinel in the extended table
        $sentinelT = Tensor::zeros($rows)->addScalarInplace($sentinel);
        // log-prob table extended with OOB sentinel at index $bins
        $oobCost = -log(1e-8);

        for ($p = 0; $p < $this->nProjections; $p++) {
            $min   = $this->binEdges[$p]['min'];
            $width = $this->binEdges[$p]['width'];

            $rawBin = $z->col($p)->addScalar(-$min)->mulScalar(1.0 / $width)->floor();

            // OOB-high clips to sentinel naturally; OOB-low is overridden via where
            $clipped = $rawBin->clip(0.0, $sentinel);          // [0, bins]: bins = sentinel index
            $oobLow  = $rawBin->lessScalar(0.0);               // 1 where val < min
            $binIdx  = $oobLow->where($sentinelT, $clipped);   // redirect below-range to sentinel

            $logProbTable = Tensor::fromArray(
                array_merge($this->histograms[$p], [$oobCost])  // append sentinel log-prob at [$bins]
            );

            $totalScore->addInplace(Tensor::gatherIndices($binIdx, $logProbTable));
            unset($rawBin, $clipped, $oobLow, $binIdx, $logProbTable);
        }

        return $totalScore->mulScalarInplace(1.0 / $this->nProjections);
    }

    public function trained(): bool
    {
        return $this->projections !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['nProjections' => $this->nProjections, 'bins' => $this->bins, 'histograms' => $this->histograms, 'binEdges' => $this->binEdges]));
        if ($this->projections !== null) {
            SafeTensorsIO::save($dir . '/model.safetensors', ['projections' => $this->projections]);
        }
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int) $c['nProjections'], (int) $c['bins']);
        $i->histograms = $c['histograms'] ?? [];
        $i->binEdges   = $c['binEdges']   ?? [];
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $t = SafeTensorsIO::load($stPath);
            $i->projections = $t['projections'] ?? null;
        }
        return $i;
    }
}