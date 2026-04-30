<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Data\DataFrame;
use Pml\Tensor;

/**
 * Categorical encoder for DataFrame STRING columns.
 *
 * Combines target encoding (smoothed mean) + frequency encoding into a
 * single fit/transform pass.  All statistics computed in C via FFI.
 *
 * Per STRING column two new FLOAT32 columns are added:
 *   "{$col}_te"   — James-Stein smoothed mean of y per category
 *   "{$col}_fe"   — category frequency (fraction of training rows)
 *
 * Fit on training data only; transform() reuses stored stats for validation/test.
 */
final class CategoricalEncoder
{
    /** col → [cat_means, global_mean, cat_freqs] */
    private array $stats = [];

    public function __construct(
        private readonly float $smoothing = 10.0
    ) {}

    /**
     * Fit statistics on training DataFrame.
     * @param string[] $cols  STRING column names to encode
     * @param Tensor   $y     [N] float32 labels (log-transformed target)
     */
    public function fit(DataFrame $df, array $cols, Tensor $y): self
    {
        $globalMean = (float) $y->mean();   // C — single scalar
        foreach ($cols as $col) {
            $catMeans = $df->targetEncodeFit($col, $y, $this->smoothing);  // C [n_cats]
            $catFreqs = $df->freqEncodeFit($col);                           // C [n_cats]
            $this->stats[$col] = [$catMeans, $globalMean, $catFreqs];
        }
        return $this;
    }

    /**
     * Add encoded columns to $df.  Unknown/missing stats are skipped.
     * Returns the augmented DataFrame.
     */
    public function transform(DataFrame $df, array $cols): DataFrame
    {
        foreach ($cols as $col) {
            if (!isset($this->stats[$col])) continue;
            if (!in_array($col, $df->columns(), true)) continue;

            [$catMeans, $globalMean, $catFreqs] = $this->stats[$col];

            // Target-encoded column (C)
            $te = $df->targetEncodeTransform($col, $catMeans, $globalMean);
            $df = $df->withTensorColumn($col . '_te', $te);

            // Frequency-encoded column (C)
            $fe = $df->freqEncodeTransform($col, $catFreqs);
            $df = $df->withTensorColumn($col . '_fe', $fe);
        }
        return $df;
    }

    /** fit() then transform() in one call (training convenience). */
    public function fitTransform(DataFrame $df, array $cols, Tensor $y): DataFrame
    {
        $this->fit($df, $cols, $y);
        return $this->transform($df, $cols);
    }

    /** Returns the column names this encoder has been fitted on. */
    public function fittedCols(): array
    {
        return array_keys($this->stats);
    }
}
