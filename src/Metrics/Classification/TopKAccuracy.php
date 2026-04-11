<?php
declare(strict_types=1);

namespace Pml\Metrics\Classification;

use Pml\Metrics\Metric;
use Pml\Tensor;

/**
 * Top-K Accuracy — fraction of samples where the true label is within the top K predictions.
 * Requires predictions to be a [N × classes] probability matrix.
 */
final class TopKAccuracy implements Metric
{
    public function __construct(private readonly int $k = 3) {}

    public function score(Tensor $predictions, ?Tensor $labels): float
    {
        if ($labels === null) {
            throw new \InvalidArgumentException("TopKAccuracy requires ground-truth labels.");
        }

        $n       = $predictions->shape()[0];
        $numCls  = $predictions->shape()[1] ?? 1;
        $kActual = min($this->k, $numCls);

        // Top-K indices per row via argsort descending
        $sorted = $predictions->argsort(1);                        // ascending → take last k
        $flatS  = $sorted->toFlatArray();
        $flatL  = $labels->toFlatArray();
        $hits   = 0;

        for ($i = 0; $i < $n; $i++) {
            $trueLabel = (int) $flatL[$i];
            for ($j = $numCls - 1; $j >= $numCls - $kActual; $j--) {
                if ((int) $flatS[$i * $numCls + $j] === $trueLabel) {
                    $hits++;
                    break;
                }
            }
        }

        return $n > 0 ? $hits / $n : 0.0;
    }

    public function range(): array { return [0.0, 1.0]; }
}
