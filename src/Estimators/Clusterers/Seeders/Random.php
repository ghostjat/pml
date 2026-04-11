<?php
declare(strict_types=1);

namespace Pml\Estimators\Clusterers\Seeders;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Random seeder — picks K rows uniformly at random without replacement.
 * O(k) FFI calls; used as a fast but non-deterministic baseline.
 *
 * JIT & Memory Optimized:
 * - Fisher-Yates partial shuffle runs in PHP; only the k chosen indices cross FFI.
 */
final class Random implements Seeder
{
    public function seed(Dataset $dataset, int $k): Tensor
    {
        $n = $dataset->numRows();

        if ($k > $n) {
            throw new \InvalidArgumentException("Cannot seed {$k} clusters from {$n} samples.");
        }

        // Partial Fisher-Yates — only shuffle first k positions
        $indices = range(0, $n - 1);
        for ($i = 0; $i < $k; $i++) {
            $j = mt_rand($i, $n - 1);
            [$indices[$i], $indices[$j]] = [$indices[$j], $indices[$i]];
        }
        $chosen = array_slice($indices, 0, $k);

        $idxT = Tensor::fromArray($chosen, Tensor::DTYPE_INT32);
        return $dataset->samples()->take($idxT, 0);                   // [k × D] zero-copy
    }
}
