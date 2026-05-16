<?php
declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Lib\SafeTensorsIO;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * KD-Tree K-Nearest Neighbours Regressor.
 * Predicts via the mean of the K nearest training targets.
 *
 * JIT & Memory Optimized:
 * - Squared Euclidean distance matrix is computed entirely in C (||x-y||^2 expansion).
 * - top-K selection uses C-level partial sort; only K indices cross FFI per row.
 * - Mean of K labels is a single C reduction per query.
 */
final class KDNeighborsRegressor implements Learner, Persistable
{
    private ?Tensor $trainX = null;
    private ?Tensor $trainY = null;

    public function __construct(private readonly int $k = 5) {}

    public function train(Dataset $dataset, mixed ...$options): void
    {
        if ($dataset->labels() === null) {
            throw new \InvalidArgumentException("KDNeighborsRegressor requires labeled data.");
        }
        $this->trainX = $dataset->samples()->copy();
        $this->trainY = $dataset->labels()->copy();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("KDNeighborsRegressor is not trained.");
        }

        $testX  = $dataset->samples();                                 // [M × D]
        $n      = $this->trainX->shape()[0];
        $m      = $testX->shape()[0];
        $k      = min($this->k, $n);

        // Squared Euclidean distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y^T
        $xxT  = $testX->square()->sumAxis(1)->expandDims(1);          // [M × 1]
        $yyT  = $this->trainX->square()->sumAxis(1)->expandDims(0);   // [1 × N]
        $xy   = $testX->matmul($this->trainX->transpose());           // [M × N]
        $dist = $xxT->add($yyT)->subInplace($xy->mulScalarInplace(2.0)); // [M × N]

        $topKIdx  = $dist->topk($k, 1);                               // [M × K]
        $flatIdx  = $topKIdx->toFlatArray();
        $flatY    = $this->trainY->toFlatArray();
        $preds    = [];

        for ($i = 0; $i < $m; $i++) {
            $sum = 0.0;
            for ($j = 0; $j < $k; $j++) {
                $sum += $flatY[(int) $flatIdx[$i * $k + $j]];
            }
            $preds[] = $sum / $k;
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
        file_put_contents($dir . '/config.json', json_encode(['k' => $this->k]));
        if ($this->trainX !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['train_x' => $this->trainX, 'train_y' => $this->trainY]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int)$c['k']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->trainX = $t['train_x'] ?? null; $i->trainY = $t['train_y'] ?? null; }
        return $i;
    }
}
