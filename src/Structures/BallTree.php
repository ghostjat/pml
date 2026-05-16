<?php

declare(strict_types=1);

namespace Pml\Structures;

use Pml\Tensor;
use Pml\Dataset;

/**
 * Ball Tree.
 * A spatial index that partitions data into nested hyper-spheres (balls) rather than axis-aligned planes.
 * Performs significantly better than KD-Trees in very high-dimensional spaces.
 */
final class BallTree
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
        $this->tree = $this->buildRecursive($indices);
    }

    private function buildRecursive(array $indices): array
    {
        if (count($indices) <= $this->leafSize) {
            return ['type' => 'leaf', 'indices' => $indices];
        }

        // 1. Calculate Centroid of current subset natively in C
        $idxT = Tensor::fromArray($indices);
        $subset = $this->samples->take($idxT, 0);
        $centroid = $subset->meanAxis(0);
        
        // 2. Find the point furthest from the centroid (Point A)
        $distToCentroid = $subset->sub($centroid)->square()->sumAxis(1);
        $idxA = $indices[$distToCentroid->argmax()];
        $pointA = $this->samples->row($idxA);

        // 3. Find the point furthest from Point A (Point B)
        $distToA = $subset->sub($pointA)->square()->sumAxis(1);
        $idxB = $indices[$distToA->argmax()];
        $pointB = $this->samples->row($idxB);

        // 4. Partition points based on closest anchor (A or B)
        $distToB = $subset->sub($pointB)->square()->sumAxis(1)->toFlatArray();
        $distToA = $distToA->toFlatArray();
        
        $leftIndices = [];
        $rightIndices = [];

        foreach ($indices as $i => $globalIdx) {
            if ($distToA[$i] < $distToB[$i]) {
                $leftIndices[] = $globalIdx;
            } else {
                $rightIndices[] = $globalIdx;
            }
        }

        // Handle edge case of duplicate overlapping points
        if (empty($leftIndices) || empty($rightIndices)) {
            return ['type' => 'leaf', 'indices' => $indices];
        }

        // 5. Calculate Radius bounds for pruning
        $radiusA = max(array_intersect_key($distToA, array_flip(array_keys($leftIndices))));
        $radiusB = max(array_intersect_key($distToB, array_flip(array_keys($rightIndices))));

        return [
            'type'    => 'node',
            'pointA'  => $pointA,
            'pointB'  => $pointB,
            'radiusA' => sqrt($radiusA),
            'radiusB' => sqrt($radiusB),
            'left'    => $this->buildRecursive($leftIndices),
            'right'   => $this->buildRecursive($rightIndices)
        ];
    }
}