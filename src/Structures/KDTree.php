<?php

declare(strict_types=1);

namespace Pml\Structures;

use Pml\Tensor;
use Pml\Dataset;

/**
 * K-Dimensional Tree (KDTree).
 * A space-partitioning data structure organizing points in a K-dimensional space.
 * Reduces nearest neighbor searches from O(N) to O(log N).
 * * JIT & Memory Optimized:
 * - Builds a hierarchical PHP array index map.
 * - Leaves fetch batched contiguous C-Tensors via `take()` for ultra-fast vector evaluation.
 */
final class KDTree
{
    private ?array $tree = null;
    private ?Tensor $samples = null;
    private ?Tensor $labels = null;
    private int $leafSize;

    public function __construct(int $leafSize = 30)
    {
        $this->leafSize = $leafSize;
    }

    public function build(Dataset $dataset): void
    {
        $this->samples = $dataset->samples();
        $this->labels = $dataset->labels();
        $n = $this->samples->shape()[0];
        
        $indices = range(0, $n - 1);
        $this->tree = $this->buildRecursive($indices, 0);
    }

    private function buildRecursive(array $indices, int $depth): array
    {
        if (count($indices) <= $this->leafSize) {
            return ['type' => 'leaf', 'indices' => $indices];
        }

        $k = $this->samples->shape()[1];
        $axis = $depth % $k;

        // Sort indices by the value of the splitting axis
        $axisCol = $this->samples->col($axis)->toFlatArray();
        usort($indices, fn($a, $b) => $axisCol[$a] <=> $axisCol[$b]);

        $medianIdx = (int) floor(count($indices) / 2);
        
        return [
            'type'  => 'node',
            'axis'  => $axis,
            'split' => $axisCol[$indices[$medianIdx]],
            'left'  => $this->buildRecursive(array_slice($indices, 0, $medianIdx), $depth + 1),
            'right' => $this->buildRecursive(array_slice($indices, $medianIdx), $depth + 1)
        ];
    }

    /**
     * Efficiently queries the K-nearest neighbors for a given test Tensor vector.
     * @return array [Tensor K-Labels, Tensor K-Distances]
     */
    public function query(Tensor $queryVector, int $k): array
    {
        $bestIndices = [];
        $bestDistances = [];

        $this->searchRecursive($this->tree, $queryVector, $k, $bestIndices, $bestDistances);
        
        $idxT = Tensor::fromArray($bestIndices);
        return [
            $this->labels->take($idxT, 0),
            Tensor::fromArray($bestDistances)
        ];
    }

    private function searchRecursive(array $node, Tensor $q, int $k, array &$bestIdx, array &$bestDist): void
    {
        if ($node['type'] === 'leaf') {
            // Leaf hit: Vectorized distance evaluation across the leaf's subset instantly in C
            $leafIndices = Tensor::fromArray($node['indices']);
            $leafSamples = $this->samples->take($leafIndices, 0);
            
            $distSq = $leafSamples->sub($q)->square()->sumAxis(1)->toFlatArray();
            
            foreach ($node['indices'] as $i => $idx) {
                $d = $distSq[$i];
                $bestDist[$idx] = $d;
            }
            
            asort($bestDist);
            $bestDist = array_slice($bestDist, 0, $k, true);
            $bestIdx = array_keys($bestDist);
            return;
        }

        $axisVal = $q->buffer()[$node['axis']]; // Direct access to C-buffer
        
        $primary = $axisVal < $node['split'] ? $node['left'] : $node['right'];
        $secondary = $axisVal < $node['split'] ? $node['right'] : $node['left'];

        $this->searchRecursive($primary, $q, $k, $bestIdx, $bestDist);

        // Check if we need to search the other side of the splitting plane
        $worstBestDist = count($bestDist) < $k ? INF : end($bestDist);
        $distToPlaneSq = ($axisVal - $node['split']) ** 2;

        if ($distToPlaneSq < $worstBestDist) {
            $this->searchRecursive($secondary, $q, $k, $bestIdx, $bestDist);
        }
    }
}