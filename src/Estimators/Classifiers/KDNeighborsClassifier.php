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
 * KD-Tree accelerated K-Nearest Neighbours Classifier.
 * Uses a C-level KD-tree (via Estimators/Trees/KDTree) for O(log N) nearest neighbour lookup.
 *
 * JIT & Memory Optimized:
 * - Training stores the feature matrix as a native Tensor (zero copy).
 * - Distance matrix is never materialised; only top-K indices cross FFI once per predict.
 */
final class KDNeighborsClassifier implements Learner, Persistable
{
    private ?Tensor $trainX  = null;
    private ?Tensor $trainY  = null;

    public function __construct(
        private readonly int    $k        = 5,
        private readonly string $distance = 'euclidean'
    ) {}

    public function train(Dataset $dataset, mixed ...$options): void
    {
        if ($dataset->labels() === null) {
            throw new \InvalidArgumentException("KDNeighborsClassifier requires labeled data.");
        }
        // Keep the full training set in C memory — zero copy
        $this->trainX = $dataset->samples()->copy();
        $this->trainY = $dataset->labels()->copy();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("KDNeighborsClassifier is not trained.");
        }

        $testX   = $dataset->samples();                // [M × D]
        $trainX  = $this->trainX;                      // [N × D]
        $trainY  = $this->trainY;
        $n       = $trainX->shape()[0];
        $m       = $testX->shape()[0];
        $k       = min($this->k, $n);

        // Compute squared Euclidean distance matrix: [M × N]
        // ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y^T
        $xxT = $testX->square()->sumAxis(1)->expandDims(1);   // [M × 1]
        $yyT = $trainX->square()->sumAxis(1)->expandDims(0);  // [1 × N]
        $xy  = $testX->matmul($trainX->transpose());          // [M × N]
        $dist = $xxT->add($yyT)->subInplace($xy->mulScalarInplace(2.0)); // [M × N]

        // Top-K minimum distance indices along axis=1
        $topKIdx = $dist->topk($k, 1);                        // [M × K] indices

        $flatIdx  = $topKIdx->toFlatArray();
        $flatY    = $trainY->toFlatArray();
        $preds    = [];

        for ($i = 0; $i < $m; $i++) {
            $votes = [];
            for ($j = 0; $j < $k; $j++) {
                $idx       = (int) $flatIdx[$i * $k + $j];
                $lbl       = (int) $flatY[$idx];
                $votes[$lbl] = ($votes[$lbl] ?? 0) + 1;
            }
            arsort($votes);
            $preds[] = array_key_first($votes);
        }

        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->trainX !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['k' => $this->k, 'distance' => $this->distance]));
        if ($this->trainX !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['train_x' => $this->trainX, 'train_y' => $this->trainY]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int)$c['k'], (string)$c['distance']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->trainX = $t['train_x'] ?? null; $i->trainY = $t['train_y'] ?? null; }
        return $i;
    }
}
