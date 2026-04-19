<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Pml\Dataset;

/**
 * Optional extension of Transformer for implementations that can fuse the
 * fit and transform passes into a single dataset scan, saving one full read.
 *
 * Contrast with the base Transformer contract, which requires two separate
 * calls (fit then transform). Implement this when computing statistics and
 * applying them can share the same iteration loop — e.g. online normalization.
 *
 * Pipeline detects this interface automatically and calls fitTransform()
 * in place of fit() + transform() during training, yielding an O(n) speedup
 * for large datasets. During inference, only transform() is called as usual.
 */
interface FitTransformable extends Transformer
{
    /**
     * Fit the transformer to the dataset and return the transformed result
     * in a single pass. Semantically equivalent to fit() + transform() but
     * may be implemented more efficiently.
     *
     * MUST leave the transformer in fitted() === true state.
     */
    public function fitTransform(Dataset $dataset): Dataset;
}
