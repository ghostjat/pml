<?php

declare(strict_types=1);

namespace Pml\Classic;

use Pml\Tensor;


/**
 * Transformer: any estimator that maps input data to a new representation.
 *
 * Mirrors sklearn's TransformerMixin contract.
 */
interface Transformer extends Estimator
{
    /**
     * Apply the fitted transformation to $X.
     *
     * @param Tensor $X  Input  [n_samples, n_features_in]
     * @return Tensor    Output [n_samples, n_features_out]
     */
    public function transform(Tensor $X): Tensor;

    /**
     * Convenience: fit to $X then immediately transform it.
     *
     * Equivalent to fit($X, $y)->transform($X) but may be more efficient.
     *
     * @param Tensor      $X  Input  [n_samples, n_features]
     * @param Tensor|null $y  Optional targets (ignored by most transformers)
     * @return Tensor         Transformed [n_samples, n_features_out]
     */
    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor;
}