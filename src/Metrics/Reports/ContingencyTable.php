<?php
declare(strict_types=1);

namespace Pml\Metrics\Reports;

use Pml\Tensor;

/**
 * Contingency Table — cross-tabulation of predicted cluster vs true class.
 * Returns a 2D PHP array [predicted_cluster][true_class] = count.
 */
final class ContingencyTable
{
    /**
     * @return array<int, array<int, int>>
     */
    public function generate(Tensor $predictions, Tensor $labels): array
    {
        $pred  = $predictions->toFlatArray();
        $true  = $labels->toFlatArray();
        $table = [];

        foreach ($pred as $i => $k) {
            $c              = (int) $true[$i];
            $k              = (int) $k;
            $table[$k][$c] = ($table[$k][$c] ?? 0) + 1;
        }

        ksort($table);
        foreach ($table as &$row) ksort($row);

        return $table;
    }
}
