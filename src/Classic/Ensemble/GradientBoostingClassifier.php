<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};
use Pml\Classic\Tree\DecisionTreeRegressor;

// ═══════════════════════════════════════════════════════════════════════════
//  GradientBoostingClassifier — sklearn.ensemble.GradientBoostingClassifier
//
//  Friedman's Gradient Boosting Machine for classification.
//  Minimises the Deviance (log-loss / cross-entropy) by fitting sequences of
//  DecisionTreeRegressor models to the pseudo-residuals of the log-loss
//  gradient.
//
//  ── Binary classification (K = 2) ─────────────────────────────────────────
//
//  Uses a single additive model F (log-odds space):
//
//    Initialise:  F_0 = log(p̄ / (1 − p̄))   where p̄ = mean(y)
//
//    Pseudo-residuals (negative gradient of log-loss):
//      r_i = y_i − σ(F_i)   where σ(x) = 1 / (1 + e^{−x})
//
//    Stage t:   fit tree_t to (X, r); update F += η · tree_t.predict(X)
//
//    predict_proba:  P(y=1|x) = σ(F(x))
//                   P(y=0|x) = 1 − σ(F(x))
//
//  ── Multi-class (K > 2) ───────────────────────────────────────────────────
//
//  K separate additive models F_0, …, F_{K-1} (one per class).
//  Softmax links each row to a probability distribution:
//
//    Initialise:  F_k,0 = log(n_k / n)   (log-prior for class k)
//
//    Softmax (with log-sum-exp stability):
//      p_{i,k} = exp(F_{i,k} − lse_i) / Σ_j exp(F_{i,j} − lse_i)
//
//    Pseudo-residuals for class k:
//      r_{i,k} = I(y_i = k) − p_{i,k}
//
//    Stage t: for each class k, fit tree_{t,k} to (X, r_k);
//             update F_{:,k} += η · tree_{t,k}.predict(X)
//
//  ── Storage layout ────────────────────────────────────────────────────────
//
//  $estimators_[t][k]:  tree at stage t for class k.
//    Binary (K_eff = 1): $estimators_[t][0] = single tree.
//    Multi-class:        $estimators_[t][k] = one tree per class k.
//
//  ── BLAS operations ───────────────────────────────────────────────────────
//
//  F update per class (each stage):
//    cblas_saxpy(n, η, tree_pred, 1, F_k, 1)    →  F_k += η · pred_k
//
//  Pseudo-residual difference (binary, each stage):
//    r ← y.clone();  cblas_saxpy(n, −1, prob, 1, r, 1)  →  r = y − prob
//
//  These reduce O(n) Python-interpreter work to single BLAS-1 C calls per
//  stage and class.
// ═══════════════════════════════════════════════════════════════════════════

final class GradientBoostingClassifier implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * $estimators_[t][k] = DecisionTreeRegressor for stage t, class k.
     * For binary K_eff=1, only index k=0 is populated per stage.
     * @var DecisionTreeRegressor[][]
     */
    public readonly array $estimators_;

    /**
     * Initial F values (log-odds / log-priors).
     * Binary: float[1] — one log-odds scalar.
     * Multi-class: float[K] — log-prior for each class.
     * @var float[]
     */
    public readonly array $init_vals_;

    /** Sorted unique class labels discovered at fit() time. @var int[] */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int               $n_estimators       Number of boosting stages.
     * @param float             $learning_rate       Shrinkage η (0, 1].
     * @param int               $max_depth           Max depth of each tree.
     * @param int               $min_samples_split   Min samples required to split a node.
     * @param float             $subsample           Fraction of training samples per stage.
     *                                               < 1.0 enables Stochastic GBM.
     * @param int|string|null   $max_features        Features per split in each tree.
     * @param ?int              $random_state        RNG seed.
     */
    public function __construct(
        private readonly int             $n_estimators      = 100,
        private readonly float           $learning_rate     = 0.1,
        private readonly int             $max_depth         = 3,
        private readonly int             $min_samples_split = 2,
        private readonly float           $subsample         = 1.0,
        private readonly int|string|null $max_features      = null,
        private readonly ?int            $random_state      = null,
    ) {
        if ($n_estimators < 1) {
            throw new \InvalidArgumentException('GradientBoostingClassifier: n_estimators must be ≥ 1.');
        }
        if ($learning_rate <= 0.0) {
            throw new \InvalidArgumentException('GradientBoostingClassifier: learning_rate must be > 0.');
        }
        if ($subsample <= 0.0 || $subsample > 1.0) {
            throw new \InvalidArgumentException('GradientBoostingClassifier: subsample must be in (0, 1].');
        }
        if ($max_depth < 1) {
            throw new \InvalidArgumentException('GradientBoostingClassifier: max_depth must be ≥ 1.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Build the gradient boosting ensemble.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Class labels [n_samples] — integer-coded.
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('GradientBoostingClassifier: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('GradientBoostingClassifier: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d] = $X->shape;
        $blas    = BlasEngine::get()->ffi;

        if ($this->random_state !== null) {
            mt_srand($this->random_state);
        }

        // ── Discover classes ───────────────────────────────────────────────
        $seen = [];
        for ($i = 0; $i < $n; $i++) {
            $seen[(int) round((float) $y->buffer[$i])] = true;
        }
        ksort($seen);
        $classes  = array_keys($seen);
        $K        = count($classes);
        $classPos = array_flip($classes);  // label → index in $classes

        // ── Resolve binary vs multi-class ──────────────────────────────────
        //
        // K_eff: number of trees per boosting stage.
        //   Binary     → K_eff = 1 (one log-odds model)
        //   Multi-class → K_eff = K (one tree per class per stage)
        $binary = ($K === 2);
        $Keff   = $binary ? 1 : $K;

        // ── Build per-class binary label tensors (for efficient residuals) ─
        //
        // yBin[k][i] = 1.0 if y[i] == classes[k] else 0.0
        // For binary: only yBin[0] (indicator for class 1) is needed.
        $yBin = [];
        if ($binary) {
            $yb = new Tensor([$n]);
            $posLabel = $classes[1];   // class "1" (second sorted class)
            for ($i = 0; $i < $n; $i++) {
                $yb->buffer[$i] = ((int) round((float) $y->buffer[$i]) === $posLabel) ? 1.0 : 0.0;
            }
            $yBin[0] = $yb;
        } else {
            for ($k = 0; $k < $K; $k++) {
                $yb    = new Tensor([$n]);
                $label = $classes[$k];
                for ($i = 0; $i < $n; $i++) {
                    $yb->buffer[$i] = ((int) round((float) $y->buffer[$i]) === $label) ? 1.0 : 0.0;
                }
                $yBin[$k] = $yb;
            }
        }

        // ── Step 1: Initialise F (log-odds / log-priors) ──────────────────
        //
        // Binary:  F_0 = log(p̄ / (1 − p̄)),  p̄ = mean(y ∈ {0,1})
        //   Clamp p̄ to [ε, 1−ε] to avoid log(0).
        //
        // Multi-class: F_{k,0} = log(n_k / n)  for each class k.
        //   This initialises F to the log-prior, so after softmax P(k) ≈ n_k/n.
        $initVals = [];
        if ($binary) {
            $sumPos = 0.0;
            for ($i = 0; $i < $n; $i++) {
                $sumPos += (float) $yBin[0]->buffer[$i];
            }
            $pBar     = max(1e-7, min(1.0 - 1e-7, $sumPos / $n));
            $initVals[0] = log($pBar / (1.0 - $pBar));
        } else {
            $counts = array_fill(0, $K, 0);
            for ($i = 0; $i < $n; $i++) {
                $counts[$classPos[(int) round((float) $y->buffer[$i])]]++;
            }
            for ($k = 0; $k < $K; $k++) {
                $initVals[$k] = log(max(1, $counts[$k]) / $n);
            }
        }

        // Running F: F[k] is a Tensor [n] for class k (or just [n] for binary)
        $F = [];
        for ($k = 0; $k < $Keff; $k++) {
            $F[$k] = Tensor::full([$n], $initVals[$k]);
        }

        // ── Step 2: Boosting loop ─────────────────────────────────────────
        $estimators = [];
        $nSub       = ($this->subsample < 1.0)
                      ? max(1, (int) ceil($n * $this->subsample))
                      : $n;
        $stochastic = ($this->subsample < 1.0);

        for ($t = 0; $t < $this->n_estimators; $t++) {

            // ── Compute probabilities from current F ──────────────────────
            if ($binary) {
                // prob[0][i] = σ(F[0][i]) for binary
                $prob = $this->sigmoidTensor($F[0], $n);
            } else {
                // prob[k][i] = softmax(F[:,i])[k]  for multi-class
                $prob = $this->softmaxTensors($F, $K, $n);
            }

            $stageTrees = [];

            for ($k = 0; $k < $Keff; $k++) {
                // ── Pseudo-residuals: r = y_k − prob[k] ──────────────────
                //
                // Clone y_k indicator into r, then saxpy(-1, prob_k, r):
                //   r[i] = yBin[k][i] − prob[k][i]
                $r = $yBin[$k]->clone();
                $blas->cblas_saxpy($n, -1.0, $prob[$k]->buffer, 1, $r->buffer, 1);

                // ── Optional stochastic subsample ─────────────────────────
                if ($stochastic) {
                    $allIdx = range(0, $n - 1);
                    for ($i = 0; $i < $nSub; $i++) {
                        $j = mt_rand($i, $n - 1);
                        [$allIdx[$i], $allIdx[$j]] = [$allIdx[$j], $allIdx[$i]];
                    }
                    $subIdx = array_slice($allIdx, 0, $nSub);

                    $Xsub = new Tensor([$nSub, $d]);
                    $rsub = new Tensor([$nSub]);
                    for ($i = 0; $i < $nSub; $i++) {
                        $src    = $subIdx[$i];
                        $srcPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$src * $d]));
                        $dstPtr = \FFI::cast('float*', \FFI::addr($Xsub->buffer[$i * $d]));
                        $blas->cblas_scopy($d, $srcPtr, 1, $dstPtr, 1);
                        $rsub->buffer[$i] = $r->buffer[$src];
                    }
                    $Xtrain = $Xsub;
                    $rtrain = $rsub;
                } else {
                    $Xtrain = $X;
                    $rtrain = $r;
                }

                // ── Fit tree to pseudo-residuals ─────────────────────────
                $tree = new DecisionTreeRegressor(
                    max_depth:         $this->max_depth,
                    min_samples_split: $this->min_samples_split,
                    max_features:      $this->max_features,
                    random_state:      ($this->random_state ?? 0) + $t * $Keff + $k,
                );
                $tree->fit($Xtrain, $rtrain);
                $stageTrees[$k] = $tree;

                // ── Update F[k] += η · tree.predict(X_full) ──────────────
                $pred = $tree->predict($X);
                $blas->cblas_saxpy($n, $this->learning_rate, $pred->buffer, 1, $F[$k]->buffer, 1);
            }

            $estimators[$t] = $stageTrees;
        }

        // ── Store fitted state ─────────────────────────────────────────────
        $this->estimators_    = $estimators;
        $this->init_vals_     = $initVals;
        $this->classes_       = $classes;
        $this->n_classes_     = $K;
        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels: argmax of predict_proba().
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    Integer class labels [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $proba = $this->predict_proba($X);
        $m     = $X->shape[0];
        $K     = $this->n_classes_;
        $out   = new Tensor([$m]);

        for ($i = 0; $i < $m; $i++) {
            $base    = $i * $K;
            $bestPos = 0;
            $bestVal = (float) $proba->buffer[$base];
            for ($k = 1; $k < $K; $k++) {
                $v = (float) $proba->buffer[$base + $k];
                if ($v > $bestVal) { $bestVal = $v; $bestPos = $k; }
            }
            $out->buffer[$i] = (float) $this->classes_[$bestPos];
        }

        return $out;
    }

    /**
     * Predict class probability distributions.
     *
     * Binary:      P(y=1|x) = σ(F(x))  →  reshaped to [n, 2]
     * Multi-class: P(y=k|x) = softmax(F(x))[k]  →  [n, K]
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples, n_classes] row-major probability matrix
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "GradientBoostingClassifier::predict_proba() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m    = $X->shape[0];
        $K    = $this->n_classes_;
        $blas = BlasEngine::get()->ffi;
        $Keff = ($K === 2) ? 1 : $K;

        // ── Reconstruct F from initial values and all tree stages ──────────
        $F = [];
        for ($k = 0; $k < $Keff; $k++) {
            $F[$k] = Tensor::full([$m], $this->init_vals_[$k]);
        }

        foreach ($this->estimators_ as $stageTrees) {
            for ($k = 0; $k < $Keff; $k++) {
                $pred = $stageTrees[$k]->predict($X);
                $blas->cblas_saxpy($m, $this->learning_rate, $pred->buffer, 1, $F[$k]->buffer, 1);
            }
        }

        // ── Convert F to probability matrix [m, K] ─────────────────────────
        $out = new Tensor([$m, $K]);

        if ($K === 2) {
            // Binary: σ(F[0]) → P(class1); 1 − σ(F[0]) → P(class0)
            for ($i = 0; $i < $m; $i++) {
                $p = 1.0 / (1.0 + exp(-(float) $F[0]->buffer[$i]));
                $out->buffer[$i * 2]     = 1.0 - $p;   // P(class=classes_[0])
                $out->buffer[$i * 2 + 1] = $p;          // P(class=classes_[1])
            }
        } else {
            // Multi-class: softmax with log-sum-exp stability
            for ($i = 0; $i < $m; $i++) {
                // Find row max for numerical stability
                $maxF = (float) $F[0]->buffer[$i];
                for ($k = 1; $k < $K; $k++) {
                    $fk = (float) $F[$k]->buffer[$i];
                    if ($fk > $maxF) { $maxF = $fk; }
                }
                // Compute softmax denominator
                $sumExp = 0.0;
                for ($k = 0; $k < $K; $k++) {
                    $sumExp += exp((float) $F[$k]->buffer[$i] - $maxF);
                }
                $base = $i * $K;
                for ($k = 0; $k < $K; $k++) {
                    $out->buffer[$base + $k] = exp((float) $F[$k]->buffer[$i] - $maxF) / $sumExp;
                }
            }
        }

        return $out;
    }

    /**
     * Accuracy score on test data.
     * Mirrors sklearn's ClassifierMixin.score().
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $pred = $this->predict($X);
        $n    = $y->size;
        $ok   = 0;
        for ($i = 0; $i < $n; $i++) {
            if ((int) round((float) $y->buffer[$i]) === (int) round((float) $pred->buffer[$i])) {
                $ok++;
            }
        }
        return $ok / $n;
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Compute element-wise sigmoid of a Tensor, returning a new Tensor.
     * Used in fit() to convert log-odds F to probabilities for binary GBM.
     *
     * @return Tensor[] [prob_tensor] (length-1 array — consistent with softmaxTensors API)
     */
    private function sigmoidTensor(Tensor $F, int $n): array
    {
        $out = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $out->buffer[$i] = 1.0 / (1.0 + exp(-(float) $F->buffer[$i]));
        }
        return [$out];   // wrap in array so caller iterates same as multi-class
    }

    /**
     * Compute row-wise softmax of K parallel score Tensors.
     * Returns float[K] Tensors where out[k][i] = P(class k | sample i).
     *
     * Uses the log-sum-exp trick (subtract row-max) for numerical stability.
     *
     * @param  Tensor[] $F  K Tensors each of shape [n]
     * @param  int      $K  Number of classes
     * @param  int      $n  Number of samples
     * @return Tensor[]     K Tensors, each [n], summing to 1.0 per sample
     */
    private function softmaxTensors(array $F, int $K, int $n): array
    {
        $out = [];
        for ($k = 0; $k < $K; $k++) {
            $out[$k] = new Tensor([$n]);
        }

        for ($i = 0; $i < $n; $i++) {
            // Row-max for stability
            $maxF = (float) $F[0]->buffer[$i];
            for ($k = 1; $k < $K; $k++) {
                $fk = (float) $F[$k]->buffer[$i];
                if ($fk > $maxF) { $maxF = $fk; }
            }
            // Softmax denominator
            $sumExp = 0.0;
            $expVals = [];
            for ($k = 0; $k < $K; $k++) {
                $ev        = exp((float) $F[$k]->buffer[$i] - $maxF);
                $expVals[] = $ev;
                $sumExp   += $ev;
            }
            for ($k = 0; $k < $K; $k++) {
                $out[$k]->buffer[$i] = $expVals[$k] / $sumExp;
            }
        }

        return $out;
    }

    private function checkFitted(): void
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException(
                'GradientBoostingClassifier is not fitted. Call fit() first.'
            );
        }
    }
}
