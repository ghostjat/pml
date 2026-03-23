<?php

declare(strict_types=1);

namespace Pml\Classic;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  Scikit-Learn API Interfaces for Pml\Classic
//
//  These three interfaces mirror the mixin/base class hierarchy of sklearn:
//
//    BaseEstimator   → fit()
//    TransformerMixin → transform(), fit_transform()
//    ClassifierMixin / RegressorMixin → predict()
//
//  PHP does not support multiple class inheritance, but multiple interface
//  implementation is fully supported.  All Pml\Classic estimators implement
//  the relevant subset of these three contracts.
//
//  API Parity targets:
//    sklearn.base.BaseEstimator          → Estimator
//    sklearn.base.TransformerMixin       → Transformer
//    sklearn.base.RegressorMixin /
//      sklearn.base.ClassifierMixin      → Predictor
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Estimator: any object that can learn from data.
 *
 * Mirrors sklearn's BaseEstimator + the fit() contract shared by all
 * sklearn estimators.  The $y parameter is null for unsupervised learners.
 */
interface Estimator
{
    /**
     * Fit the model to training data.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Target vector   [n_samples] — null for unsupervised
     * @return static         Returns $this for method chaining (sklearn convention)
     */
    public function fit(Tensor $X, ?Tensor $y = null): static;
}



