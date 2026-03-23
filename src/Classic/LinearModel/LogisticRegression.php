<?php

declare(strict_types=1);

namespace Pml\Classic\LinearModel;

use Pml\{Tensor, BlasEngine, Ops};
use Pml\Classic\{Estimator, Predictor};
use Pml\Training\{AdamW, CrossEntropyLoss};

// ═══════════════════════════════════════════════════════════════════════════
//  LogisticRegression — sklearn.linear_model.LogisticRegression
//
//  Handles both binary and multinomial (softmax) classification automatically,
//  matching sklearn's solver='lbfgs' multi_class='auto' default behaviour.
//
//  ── BINARY PATH (n_classes == 2) ─────────────────────────────────────────
//
//    Linear → Sigmoid → Binary Cross-Entropy Loss
//
//    W [n, 1], b [1]
//    logits [m, 1] = X @ W + b
//    proba  [m, 1] = σ(logits)
//    loss   [1]    = BCE(proba, y)
//
//    predict_proba returns [m, 2]: [[1−p, p], ...]
//    predict      thresholds P(class=1) at 0.5
//
//  ── MULTINOMIAL PATH (n_classes > 2) ─────────────────────────────────────
//
//    Linear → (fused Softmax + Cross-Entropy Loss)
//
//    W [n, K], b [K]
//    logits [m, K] = X @ W + b
//    loss   [1]    = CrossEntropyLoss(logits, y)   ← fused softmax+NLL
//
//    predict_proba returns [m, K]: softmax probabilities per class
//    predict      returns argmax across K class columns
//
//  Both paths share the same AdamW optimiser, Xavier weight init, mini-batch
//  sampling, gradient clipping, and early-stopping logic.
//
//  Exposed attributes (sklearn naming convention):
//    coef_      [1, n_features]  (binary)  or  [n_classes, n_features]  (multi)
//    intercept_ [1]              (binary)  or  [n_classes]              (multi)
//    n_classes_ int  — number of unique target classes detected at fit time
// ═══════════════════════════════════════════════════════════════════════════

final class LogisticRegression implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────
    /** @var Tensor  [1, n_features] (binary) or [n_classes, n_features] (multi) */
    public readonly Tensor $coef_;

    /** @var Tensor  [1] (binary) or [n_classes] (multi) */
    public readonly Tensor $intercept_;

    public readonly int $n_features_in_;
    public readonly int $n_iter_;

    /** Number of unique target classes detected during fit(). */
    public readonly int $n_classes_;

    // ── Internal learnable parameters (retained for predict_proba) ────────
    private Tensor $W_;   // [n, 1] binary  |  [n, K] multiclass
    private Tensor $b_;   // [1]   binary  |  [K]    multiclass

    /**
     * @param float $C             Inverse regularisation strength (1/λ). Higher = less regularisation.
     * @param int   $max_iter      Maximum AdamW steps
     * @param float $tol           Early-stop threshold on loss change
     * @param float $learning_rate AdamW learning rate
     * @param int   $batch_size    Mini-batch size (0 = full batch)
     * @param int   $random_state  Seed for reproducible weight init and mini-batch sampling
     */
    public function __construct(
        private readonly float $C             = 1.0,
        private readonly int   $max_iter      = 100,
        private readonly float $tol           = 1e-4,
        private readonly float $learning_rate = 1e-2,
        private readonly int   $batch_size    = 0,
        private readonly int   $random_state  = 0,
    ) {}

    // ── Estimator ─────────────────────────────────────────────────────────

    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('LogisticRegression::fit() requires target tensor $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('LogisticRegression::fit() requires 2D feature matrix X.');
        }

        [$m, $n] = $X->shape;

        // ── Detect number of classes ───────────────────────────────────────
        //
        // Scan y once to find max class label. For 0-indexed labels {0,1,...,K-1}
        // this equals K = max(y) + 1 — identical to sklearn's _check_multi_class().
        //
        // We do NOT build a label→index mapping; we trust that the caller supplies
        // contiguous 0-based integer labels, matching the sklearn contract.
        $maxClass = 0;
        for ($i = 0; $i < $m; $i++) {
            $c = (int)(float)$y->buffer[$i];
            if ($c > $maxClass) {
                $maxClass = $c;
            }
        }
        $K = $maxClass + 1;   // total number of distinct classes

        // Store for use in predict() / predict_proba() dispatch
        $this->n_classes_ = $K;

        if ($this->random_state !== 0) {
            mt_srand($this->random_state);
        }

        // Weight decay = 1/C  (sklearn's C ≡ inverse regularisation strength)
        $weightDecay = 1.0 / max(1e-9, $this->C);
        $batchSize   = $this->batch_size > 0 ? $this->batch_size : $m;
        $prevLoss    = INF;
        $actualIter  = 0;

        // ════════════════════════════════════════════════════════════════════
        //  ROUTER: binary (K==2)  vs.  multinomial (K>2)
        // ════════════════════════════════════════════════════════════════════

        if ($K === 2) {
            // ── BINARY PATH ──────────────────────────────────────────────────
            //
            // Architecture: W [n, 1], b [1]
            //   logits = X @ W + b   [m, 1]
            //   proba  = σ(logits)   [m, 1]
            //   loss   = BCE(proba, y_col)
            //
            // This path is identical to the original implementation.

            $W = Tensor::xavierInit([$n, 1]);
            $b = Tensor::zeros([1]);
            $W->requiresGrad = true;
            $b->requiresGrad = true;

            $optimizer = new AdamW([$W, $b], lr: $this->learning_rate, weightDecay: $weightDecay);

            // Reshape y to column vector [m, 1] for broadcasting against logits [m, 1]
            $yCol = $y->reshape([$m, 1]);

            for ($step = 1; $step <= $this->max_iter; $step++) {
                $actualIter = $step;

                // ── Mini-batch sampling ───────────────────────────────────
                if ($batchSize < $m) {
                    $indices = array_rand(range(0, $m - 1), $batchSize);
                    if (!is_array($indices)) $indices = [$indices];
                    $Xb = $this->gatherRows($X, $indices, $batchSize, $n);
                    $yb = $this->gatherRows($yCol, $indices, $batchSize, 1);
                    $mb = $batchSize;
                } else {
                    $Xb = $X;
                    $yb = $yCol;
                    $mb = $m;
                }

                // ── Forward ───────────────────────────────────────────────
                $logits = Ops::matmul($Xb, $W);              // [mb, 1]
                $logits = self::addBiasDiff($logits, $b);    // [mb, 1]
                $proba  = self::sigmoidDiff($logits);        // [mb, 1]
                $loss   = self::bceLossDiff($proba, $yb);   // [1]

                // ── Backward + optimise ───────────────────────────────────
                $optimizer->zeroGrad();
                $loss->backward();
                AdamW::clipGradNorm([$W, $b], 5.0);
                $optimizer->step();

                // ── Early stopping ────────────────────────────────────────
                $lossVal = (float)$loss->buffer[0];
                if (abs($prevLoss - $lossVal) < $this->tol) {
                    break;
                }
                $prevLoss = $lossVal;
            }

            // ── Store fitted attributes (binary) ─────────────────────────
            $this->W_ = $W;
            $this->b_ = $b;

            // coef_ [1, n_features] — sklearn 2D convention for binary classifiers
            $coef2d = new Tensor([1, $n]);
            \FFI::memcpy($coef2d->buffer, $W->buffer, $n * 4);
            $this->coef_      = $coef2d;
            $this->intercept_ = new Tensor([1]);
            $this->intercept_->buffer[0] = (float)$b->buffer[0];

        } else {
            // ── MULTINOMIAL PATH ─────────────────────────────────────────────
            //
            // Architecture: W [n, K], b [K]
            //   logits = X @ W + b       [m, K]    — one logit column per class
            //   loss   = CrossEntropyLoss(logits, targets)
            //                              [1]      — fused softmax + NLL
            //
            // CrossEntropyLoss::forward() computes the analytical gradient
            // dL/dlogits = (softmax(logits) − one_hot(y)) / m in a single pass,
            // which is both numerically stable and more efficient than composing
            // a separate Softmax op + NLL op through the autograd tape.

            $W = Tensor::xavierInit([$n, $K]);
            $b = Tensor::zeros([$K]);
            $W->requiresGrad = true;
            $b->requiresGrad = true;

            $optimizer = new AdamW([$W, $b], lr: $this->learning_rate, weightDecay: $weightDecay);

            // Pre-convert y Tensor → PHP int[] once (CrossEntropyLoss takes int[])
            $allTargets = [];
            for ($i = 0; $i < $m; $i++) {
                $allTargets[$i] = (int)(float)$y->buffer[$i];
            }

            for ($step = 1; $step <= $this->max_iter; $step++) {
                $actualIter = $step;

                // ── Mini-batch sampling ───────────────────────────────────
                if ($batchSize < $m) {
                    $indices = array_rand(range(0, $m - 1), $batchSize);
                    if (!is_array($indices)) $indices = [$indices];
                    $Xb = $this->gatherRows($X, $indices, $batchSize, $n);
                    // Gather targets as PHP int[] — CrossEntropyLoss expects int[]
                    $tb = [];
                    foreach ($indices as $dstIdx => $srcRow) {
                        $tb[$dstIdx] = $allTargets[$srcRow];
                    }
                    $mb = $batchSize;
                } else {
                    $Xb = $X;
                    $tb = $allTargets;
                    $mb = $m;
                }

                // ── Forward ───────────────────────────────────────────────
                //
                // logits [mb, K] = Xb @ W
                $logits = Ops::matmul($Xb, $W);

                // logits [mb, K] += b [K]   (broadcast bias over the batch dimension)
                // addBiasMatDiff registers the backward closure for db[k] = Σ_i grad[i,k]
                $logits = self::addBiasMatDiff($logits, $b, $mb, $K);

                // loss [1] = fused softmax + NLL cross-entropy
                // CrossEntropyLoss::forward() sets logits._backward with the
                // analytical gradient dL/dlogits = (softmax − one_hot) / mb
                $criterion = new CrossEntropyLoss();
                $loss = $criterion->forward($logits, $tb);

                // ── Backward + optimise ───────────────────────────────────
                $optimizer->zeroGrad();
                $loss->backward();
                AdamW::clipGradNorm([$W, $b], 5.0);
                $optimizer->step();

                // ── Early stopping ────────────────────────────────────────
                $lossVal = (float)$loss->buffer[0];
                if (abs($prevLoss - $lossVal) < $this->tol) {
                    break;
                }
                $prevLoss = $lossVal;
            }

            // ── Store fitted attributes (multiclass) ──────────────────────
            $this->W_ = $W;
            $this->b_ = $b;

            // coef_ [K, n_features] — sklearn stores one row per class.
            // W is [n, K], so we transpose: coef_[k, j] = W[j, k].
            $coef = new Tensor([$K, $n]);
            for ($k = 0; $k < $K; $k++) {
                for ($j = 0; $j < $n; $j++) {
                    $coef->buffer[$k * $n + $j] = (float)$W->buffer[$j * $K + $k];
                }
            }
            $this->coef_ = $coef;

            // intercept_ [K] — one bias per class (copy from b [K])
            $intercept = new Tensor([$K]);
            \FFI::memcpy($intercept->buffer, $b->buffer, $K * 4);
            $this->intercept_ = $intercept;
        }

        // ── Shared post-fit assignments ────────────────────────────────────
        $this->n_features_in_ = $n;
        $this->n_iter_        = $actualIter;

        return $this;
    }

    // ── Predictor ─────────────────────────────────────────────────────────

    public function predict(Tensor $X): Tensor
    {
        $proba = $this->predict_proba($X);
        $m     = $proba->shape[0];
        $K     = $this->n_classes_;
        $out   = new Tensor([$m]);

        if ($K === 2) {
            // ── Binary: threshold P(class=1) ≥ 0.5 ───────────────────────
            // proba is [m, 2]; column 1 = P(class=1)
            for ($i = 0; $i < $m; $i++) {
                $out->buffer[$i] = ((float)$proba->buffer[$i * 2 + 1]) >= 0.5 ? 1.0 : 0.0;
            }
        } else {
            // ── Multiclass: argmax across K probability columns ───────────
            // proba is [m, K]; pick the column with highest softmax probability
            for ($i = 0; $i < $m; $i++) {
                $best    = 0;
                $bestVal = (float)$proba->buffer[$i * $K];
                for ($k = 1; $k < $K; $k++) {
                    $v = (float)$proba->buffer[$i * $K + $k];
                    if ($v > $bestVal) {
                        $bestVal = $v;
                        $best    = $k;
                    }
                }
                $out->buffer[$i] = (float)$best;
            }
        }

        return $out;
    }

    /**
     * Probability estimates.
     *
     * Binary    → returns [n_samples, 2]:  [[P(0), P(1)], ...]
     * Multiclass → returns [n_samples, K]: softmax probabilities for each class
     *
     * Mirrors sklearn's LogisticRegression.predict_proba() output format.
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        [$m, $n] = $X->shape;
        $K       = $this->n_classes_;

        // Inference: detach parameters so no grad tape is built
        $Wd = $this->W_->detach();
        $bd = $this->b_->detach();

        // logits [m, 1] or [m, K] = X @ W
        $logits = Ops::matmul($X, $Wd);

        if ($K === 2) {
            // ── Binary: add scalar bias, apply sigmoid ────────────────────
            $bVal = (float)$bd->buffer[0];
            for ($i = 0; $i < $m; $i++) {
                $logits->buffer[$i] += $bVal;
            }

            // Build [m, 2]: [P(class=0), P(class=1)]
            $out = new Tensor([$m, 2]);
            for ($i = 0; $i < $m; $i++) {
                $p = 1.0 / (1.0 + exp(-(float)$logits->buffer[$i]));
                $out->buffer[$i * 2]     = 1.0 - $p;  // P(class = 0)
                $out->buffer[$i * 2 + 1] = $p;         // P(class = 1)
            }
        } else {
            // ── Multiclass: add per-class bias, apply row-wise softmax ────
            for ($i = 0; $i < $m; $i++) {
                for ($k = 0; $k < $K; $k++) {
                    $logits->buffer[$i * $K + $k] += (float)$bd->buffer[$k];
                }
            }

            // Numerically stable softmax: subtract row-max before exp()
            $out = new Tensor([$m, $K]);
            for ($i = 0; $i < $m; $i++) {
                // Pass 1: find max logit in this row (for numerical stability)
                $maxLogit = -INF;
                for ($k = 0; $k < $K; $k++) {
                    $v = (float)$logits->buffer[$i * $K + $k];
                    if ($v > $maxLogit) {
                        $maxLogit = $v;
                    }
                }
                // Pass 2: compute exp(logit − max) and accumulate partition sum
                $sum = 0.0;
                for ($k = 0; $k < $K; $k++) {
                    $e = exp((float)$logits->buffer[$i * $K + $k] - $maxLogit);
                    $out->buffer[$i * $K + $k] = $e;
                    $sum += $e;
                }
                // Pass 3: normalise by partition sum → valid probability distribution
                for ($k = 0; $k < $K; $k++) {
                    $out->buffer[$i * $K + $k] /= $sum;
                }
            }
        }

        return $out;
    }

    /**
     * Accuracy score on (X, y).
     * Mirrors sklearn's score() method.
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $pred    = $this->predict($X);
        $m       = $y->size;
        $correct = 0;
        for ($i = 0; $i < $m; $i++) {
            if ((int)$pred->buffer[$i] === (int)$y->buffer[$i]) {
                $correct++;
            }
        }
        return $correct / $m;
    }

    // ── Differentiable primitives (private, used during fit) ──────────────

    /**
     * Differentiable scalar bias broadcast-add for binary path.
     *
     * Forward:  out[i] = x[i] + b[0]   for all i
     * Backward:
     *   dx[i] += grad[i]                (identity — gradient passes through)
     *   db[0] += Σ_i grad[i]            (sum over batch dimension)
     */
    private static function addBiasDiff(Tensor $x, Tensor $b): Tensor
    {
        $out  = $x->clone();
        $bVal = (float)$b->buffer[0];
        for ($i = 0; $i < $out->size; $i++) {
            $out->buffer[$i] += $bVal;
        }

        if ($x->requiresGrad || $b->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$x, $b];
            $out->_backward    = static function() use ($x, $b, $out): void {
                $blas = BlasEngine::get()->ffi;

                if ($x->requiresGrad) {
                    $x->initGrad();
                    $blas->cblas_saxpy($x->size, 1.0, $out->grad, 1, $x->grad, 1);
                }

                if ($b->requiresGrad) {
                    $b->initGrad();
                    $sum = 0.0;
                    for ($i = 0; $i < $out->size; $i++) {
                        $sum += (float)$out->grad[$i];
                    }
                    $b->grad[0] += $sum;
                }
            };
        }

        return $out;
    }

    /**
     * Differentiable bias add for multinomial path: logits [m, K] + b [K].
     *
     * Forward:  out[i, k] = x[i, k] + b[k]
     * Backward:
     *   dx[i, k] += grad[i, k]           (identity)
     *   db[k]    += Σ_i grad[i, k]       (column-wise sum over batch dimension)
     *
     * The column-wise reduction for db mirrors the shape of the bias: each
     * class has its own bias scalar that receives gradient from all m samples.
     */
    private static function addBiasMatDiff(Tensor $x, Tensor $b, int $m, int $K): Tensor
    {
        $out = $x->clone();
        for ($i = 0; $i < $m; $i++) {
            for ($k = 0; $k < $K; $k++) {
                $out->buffer[$i * $K + $k] += (float)$b->buffer[$k];
            }
        }

        if ($x->requiresGrad || $b->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$x, $b];
            $out->_backward    = static function() use ($x, $b, $out, $m, $K): void {
                if ($x->requiresGrad) {
                    $x->initGrad();
                    $blas = BlasEngine::get()->ffi;
                    // dx = grad  (identity: gradient flows through unchanged)
                    $blas->cblas_saxpy($x->size, 1.0, $out->grad, 1, $x->grad, 1);
                }

                if ($b->requiresGrad) {
                    $b->initGrad();
                    // db[k] = Σ_i grad[i, k]  — column-wise reduction
                    for ($k = 0; $k < $K; $k++) {
                        $sum = 0.0;
                        for ($i = 0; $i < $m; $i++) {
                            $sum += (float)$out->grad[$i * $K + $k];
                        }
                        $b->grad[$k] += $sum;
                    }
                }
            };
        }

        return $out;
    }

    /**
     * Differentiable sigmoid activation.
     *
     * Forward:  s[i] = 1 / (1 + exp(−x[i]))
     * Backward: dx[i] += grad[i] · s[i] · (1 − s[i])
     *
     * Note: we store s (not x) in the closure — s·(1−s) = dσ/dx is cheaper
     * to recompute from the already-computed s than to re-apply exp(−x).
     */
    private static function sigmoidDiff(Tensor $x): Tensor
    {
        $out = new Tensor($x->shape);
        for ($i = 0; $i < $x->size; $i++) {
            $out->buffer[$i] = 1.0 / (1.0 + exp(-(float)$x->buffer[$i]));
        }

        if ($x->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$x];
            $out->_backward    = static function() use ($x, $out): void {
                $x->initGrad();
                for ($i = 0; $i < $x->size; $i++) {
                    $s = (float)$out->buffer[$i];   // σ(x_i) — already computed
                    $x->grad[$i] += (float)$out->grad[$i] * $s * (1.0 - $s);
                }
            };
        }

        return $out;
    }

    /**
     * Differentiable Binary Cross-Entropy loss (returns scalar Tensor [1]).
     *
     * Forward:  L = −(1/m) · Σ_i [ y_i · log(p_i + ε) + (1−y_i) · log(1−p_i + ε) ]
     *
     * Backward: dp[i] += dL · (−y_i/(p_i+ε) + (1−y_i)/(1−p_i+ε)) / m
     *
     * The ε = 1e-7 clamp prevents log(0) — matches PyTorch's BCELoss clamping.
     * y is treated as a constant (no grad required).
     */
    private static function bceLossDiff(Tensor $proba, Tensor $y): Tensor
    {
        $m   = $proba->size;
        $eps = 1e-7;
        $loss = 0.0;

        for ($i = 0; $i < $m; $i++) {
            $p    = max(min((float)$proba->buffer[$i], 1.0 - $eps), $eps);
            $yi   = (float)$y->buffer[$i];
            $loss += -$yi * log($p) - (1.0 - $yi) * log(1.0 - $p);
        }
        $loss /= $m;

        $out           = new Tensor([1]);
        $out->buffer[0] = $loss;

        if ($proba->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$proba];
            $out->_backward    = static function() use ($proba, $y, $out, $m, $eps): void {
                $proba->initGrad();
                $dLoss = (float)$out->grad[0];

                for ($i = 0; $i < $m; $i++) {
                    $p  = max(min((float)$proba->buffer[$i], 1.0 - $eps), $eps);
                    $yi = (float)$y->buffer[$i];
                    // dL/dp_i = (−y_i/p_i + (1−y_i)/(1−p_i)) / m
                    $dp = (-$yi / $p + (1.0 - $yi) / (1.0 - $p)) / $m;
                    $proba->grad[$i] += $dLoss * $dp;
                }
            };
        }

        return $out;
    }

    // ── Utility ───────────────────────────────────────────────────────────

    /**
     * Gather specified rows from a 2D tensor into a new [batchSize, cols] tensor.
     * Uses cblas_scopy per row — O(batchSize) FFI calls each doing O(cols) C work.
     */
    private function gatherRows(Tensor $X, array $indices, int $batchSize, int $cols): Tensor
    {
        $blas = BlasEngine::get()->ffi;
        $out  = new Tensor([$batchSize, $cols]);

        foreach ($indices as $dstRow => $srcRow) {
            $src = \FFI::cast('float*', \FFI::addr($X->buffer[$srcRow * $cols]));
            $dst = \FFI::cast('float*', \FFI::addr($out->buffer[$dstRow * $cols]));
            $blas->cblas_scopy($cols, $src, 1, $dst, 1);
        }

        return $out;
    }

    private function checkFitted(): void
    {
        if (!isset($this->W_)) {
            throw new \RuntimeException('LogisticRegression is not fitted. Call fit() first.');
        }
    }
}
