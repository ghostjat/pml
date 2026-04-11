<?php
declare(strict_types=1);

namespace Pml\Estimators\Clusterers\Seeders;

use Pml\Tensor;
use Pml\Dataset;

/**
 * KMC2 — Markov Chain Monte Carlo k-means++ seeding approximation.
 * O(k * m) time vs O(k * N) for full k-means++, where m << N.
 *
 * JIT & Memory Optimized:
 * - Distance computation stays in C; PHP only reads one scalar per chain step.
 * - The chain is short (m steps), so FFI overhead is negligible.
 */
final class KMC2 implements Seeder
{
    public function __construct(private readonly int $m = 200) {}

    public function seed(Dataset $dataset, int $k): Tensor
    {
        $x    = $dataset->samples();
        $n    = $dataset->numRows();
        $d    = $dataset->numColumns();
        $m    = min($this->m, $n);

        // Pick the first center uniformly at random
        $seedRows = [mt_rand(0, $n - 1)];

        for ($i = 1; $i < $k; $i++) {
            // Sample a random initial candidate
            $q = $this->rowDistance($x, $n, $d, mt_rand(0, $n - 1), $seedRows);

            // Run a short Markov chain of length m
            for ($j = 0; $j < $m - 1; $j++) {
                $candidate = mt_rand(0, $n - 1);
                $dCandidate = $this->rowDistance($x, $n, $d, $candidate, $seedRows);
                if ($dCandidate > 0.0 && ($q === 0.0 || (mt_rand() / mt_getrandmax()) < $dCandidate / $q)) {
                    $q = $dCandidate;
                    $candidate = $candidate;  // accepted
                }
            }

            $seedRows[] = $candidate ?? mt_rand(0, $n - 1);
        }

        // Stack the seed rows into a [k × D] tensor
        $idxT   = Tensor::fromArray($seedRows, Tensor::DTYPE_INT32);
        return $x->take($idxT, 0);
    }

    /**
     * Minimum squared Euclidean distance from row $rowIdx to the nearest existing seed.
     */
    private function rowDistance(Tensor $x, int $n, int $d, int $rowIdx, array $seedRows): float
    {
        $row  = $x->row($rowIdx);
        $minD = INF;

        foreach ($seedRows as $seedIdx) {
            $seed  = $x->row($seedIdx);
            $diff  = $row->sub($seed);
            $dist  = $diff->dot($diff);
            if ($dist < $minD) $minD = $dist;
        }

        return $minD === INF ? 0.0 : $minD;
    }
}
