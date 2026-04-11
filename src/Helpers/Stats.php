<?php
declare(strict_types=1);

namespace Pml\Helpers;

use Pml\Tensor;

/**
 * Native statistical helper functions — all computations routed through C-level Tensor ops.
 * Zero PHP-loop overhead: every reduction crosses the FFI boundary exactly once.
 */
final class Stats
{
    /**
     * Mean of a flat 1-D tensor.
     */
    public static function mean(Tensor $t): float
    {
        return $t->mean();
    }

    /**
     * Population variance using the C-level parallel reduction.
     */
    public static function variance(Tensor $t): float
    {
        return $t->variance();
    }

    /**
     * Standard deviation.
     */
    public static function std(Tensor $t): float
    {
        return $t->std();
    }

    /**
     * Median via C-level partial sort.
     */
    public static function median(Tensor $t): float
    {
        return $t->median();
    }

    /**
     * Percentile (0-100). Routes through C sort + index arithmetic.
     */
    public static function percentile(Tensor $t, float $p): float
    {
        if ($p < 0.0 || $p > 100.0) {
            throw new \InvalidArgumentException("Percentile must be between 0 and 100.");
        }
        $sorted = $t->sort();
        $n      = $sorted->size();
        $idx    = (int) round(($p / 100.0) * ($n - 1));
        $arr    = $sorted->toFlatArray();
        return $arr[$idx];
    }

    /**
     * Inter-quartile range.
     */
    public static function iqr(Tensor $t): float
    {
        return self::percentile($t, 75.0) - self::percentile($t, 25.0);
    }

    /**
     * Column-wise means: returns a 1-D Tensor of shape [features].
     */
    public static function columnMeans(Tensor $x): Tensor
    {
        return $x->meanAxis(0);
    }

    /**
     * Column-wise standard deviations.
     */
    public static function columnStds(Tensor $x): Tensor
    {
        $mean    = $x->meanAxis(0);
        $sq      = $x->square()->meanAxis(0);
        $variance = $sq->sub($mean->square());
        return $variance->sqrt()->clip(1e-8, INF);
    }
}
