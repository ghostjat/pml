<?php
declare(strict_types=1);

namespace Pml\Metrics\Reports;

use Pml\Tensor;

/**
 * Error Analysis — per-sample residuals report for regression tasks.
 * Returns descriptive statistics over the absolute error distribution.
 *
 * JIT & Memory Optimized: all reductions are single C calls.
 */
final class ErrorAnalysis
{
    /**
     * @return array{mean: float, median: float, std: float, min: float, max: float, mae: float, rmse: float}
     */
    public function generate(Tensor $predictions, Tensor $labels): array
    {
        $errors = $predictions->sub($labels);
        $abs    = $errors->abs();

        return [
            'mean'   => $errors->mean(),
            'median' => $errors->median(),
            'std'    => $errors->std(),
            'min'    => $errors->min(),
            'max'    => $errors->max(),
            'mae'    => $abs->mean(),
            'rmse'   => sqrt($errors->square()->mean()),
        ];
    }
}
