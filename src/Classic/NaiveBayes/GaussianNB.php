<?php

declare(strict_types=1);

namespace Pml\Classic\NaiveBayes;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  GaussianNB — sklearn.naive_bayes.GaussianNB
//
//  Gaussian Naïve Bayes classifier.  Assumes each feature is independently
//  Gaussian-distributed within each class:
//
//    P(x_j | c) = N(x_j; θ_{c,j}, σ²_{c,j})
//               = 1/√(2π σ²_{c,j}) · exp(−(x_j − θ_{c,j})² / (2 σ²_{c,j}))
//
//  Classification rule (Bayes' theorem):
//
//    ĉ = argmax_c  log P(c) + Σ_j log P(x_j | c)
//
//  ── Fit ─────────────────────────────────────────────────────────────────
//
//  For each class c in {0..K-1}, gather the n_c rows where y == c:
//
//    θ_{c,j}  = (1/n_c) · Σ_{i: y_i=c} X_{i,j}          (class mean)
//    σ²_{c,j} = (1/n_c) · Σ_{i: y_i=c} (X_{i,j} − θ)²   (class variance)
//    P(c)     = n_c / n                                   (class prior)
//
//  Variance smoothing (sklearn var_smoothing):
//    A global epsilon prevents singular distributions when a feature has
//    zero within-class variance:
//
//      σ²_{c,j} += ε,   ε = var_smoothing · max_j( Var_global(X_j) )
//
//    where Var_global(X_j) = (1/n) Σ_i (X_{i,j} − X̄_j)² across ALL samples.
//
//  ── Predict (log-space Gaussian PDF) ────────────────────────────────────
//
//  Evaluating the product Π_j P(x_j | c) directly would underflow for large
//  d.  All arithmetic is done in log-space:
//
//    logLik_{c,j} = −0.5 · log(2π σ²_{c,j}) − (x_j − θ_{c,j})² / (2 σ²_{c,j})
//
//  Summing over features:
//    logJoint[c] = log P(c) + Σ_j logLik_{c,j}
//
//  Softmax (log-sum-exp) to convert to probabilities without overflow:
//    maxLog      = max_c logJoint[c]
//    log_denom   = maxLog + log(Σ_c exp(logJoint[c] − maxLog))
//    P(c | x)   = exp(logJoint[c] − log_denom)
//
//  ── Storage ─────────────────────────────────────────────────────────────
//
//  theta_        [n_classes, n_features] — per-class feature means
//  var_          [n_classes, n_features] — per-class feature variances (+ ε)
//  class_prior_  [n_classes]             — prior probabilities P(c)
//  class_count_  [n_classes]             — sample counts per class
//
//  All four are stored as flat 1D Float32 Tensors.  Row-major layout:
//    theta_->buffer[c * n_features + j]  — mean of class c, feature j
// ═══════════════════════════════════════════════════════════════════════════

final class GaussianNB implements Estimator, Predictor
{
    // ── Fitted attributes (sklearn naming convention) ─────────────────────

    /** @var Tensor  Class means [n_classes, n_features] (flat 1D) */
    public readonly Tensor $theta_;

    /** @var Tensor  Class variances [n_classes, n_features] (flat 1D, smoothed) */
    public readonly Tensor $var_;

    /** @var Tensor  Prior P(c) [n_classes] */
    public readonly Tensor $class_prior_;

    /** @var Tensor  Sample counts per class [n_classes] */
    public readonly Tensor $class_count_;

    /** Unique class labels sorted ascending. @var int[] */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    /**
     * @param float $var_smoothing  Portion of the largest feature variance added to
     *                              all variances for numerical stability.  Default 1e-9
     *                              matches sklearn's GaussianNB(var_smoothing=1e-9).
     */
    public function __construct(
        private readonly float $var_smoothing = 1e-9,
    ) {}

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Compute per-class means, variances, and priors from training data.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Integer class labels [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('GaussianNB::fit() requires target $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('GaussianNB::fit() requires a 2-D feature matrix X.');
        }

        [$n, $d] = $X->shape;

        // ── Discover unique class labels ───────────────────────────────────
        $seen = [];
        for ($i = 0; $i < $n; $i++) {
            $seen[(int) round((float) $y->buffer[$i])] = true;
        }
        ksort($seen);
        $classes    = array_keys($seen);
        $nC         = count($classes);
        $classToPos = array_flip($classes);   // label → index in $classes

        // ── Allocate fitted-attribute tensors (all zero-initialised) ───────
        $theta       = new Tensor([$nC, $d]);   // means
        $var         = new Tensor([$nC, $d]);   // variances
        $classCount  = new Tensor([$nC]);       // n_c
        $classPrior  = new Tensor([$nC]);       // P(c)

        // ── Accumulate per-class sums: Σ X_{i,j}  for i where y_i = c ─────
        //
        // Two-pass algorithm:
        //   Pass 1 — accumulate sums and counts
        //   Pass 2 — compute means, then accumulate squared deviations
        //
        // PHP loops are unavoidable here: samples belonging to a class are
        // scattered (not contiguous), so no BLAS column-stride operation applies.
        for ($i = 0; $i < $n; $i++) {
            $lbl = (int) round((float) $y->buffer[$i]);
            $c   = $classToPos[$lbl];
            $classCount->buffer[$c] = (float) $classCount->buffer[$c] + 1.0;

            $base = $c * $d;
            $off  = $i * $d;
            for ($j = 0; $j < $d; $j++) {
                $theta->buffer[$base + $j] =
                    (float) $theta->buffer[$base + $j] + (float) $X->buffer[$off + $j];
            }
        }

        // ── Pass 1b: divide sums by counts to get means θ_{c,j} ───────────
        for ($c = 0; $c < $nC; $c++) {
            $nc   = (float) $classCount->buffer[$c];
            $base = $c * $d;
            if ($nc > 0.0) {
                for ($j = 0; $j < $d; $j++) {
                    $theta->buffer[$base + $j] = (float) $theta->buffer[$base + $j] / $nc;
                }
            }
        }

        // ── Pass 2: accumulate squared deviations from the class mean ──────
        //
        //   var_{c,j} = (1/n_c) · Σ_{i: y_i=c} (X_{i,j} − θ_{c,j})²
        for ($i = 0; $i < $n; $i++) {
            $lbl  = (int) round((float) $y->buffer[$i]);
            $c    = $classToPos[$lbl];
            $base = $c * $d;
            $off  = $i * $d;

            for ($j = 0; $j < $d; $j++) {
                $diff = (float) $X->buffer[$off + $j] - (float) $theta->buffer[$base + $j];
                $var->buffer[$base + $j] =
                    (float) $var->buffer[$base + $j] + $diff * $diff;
            }
        }

        // Divide accumulated squared deviations by class count → variance
        for ($c = 0; $c < $nC; $c++) {
            $nc   = (float) $classCount->buffer[$c];
            $base = $c * $d;
            if ($nc > 0.0) {
                for ($j = 0; $j < $d; $j++) {
                    $var->buffer[$base + $j] = (float) $var->buffer[$base + $j] / $nc;
                }
            }
        }

        // ── Variance smoothing: ε = var_smoothing · max(global feature var) ─
        //
        // "Global feature variance" = Var over ALL samples (not per class).
        // sklearn computes np.var(X, axis=0).max() on the full training matrix,
        // then adds var_smoothing * that_max to every entry of var_.
        //
        // We compute the global variance per feature in an O(n·d) PHP loop,
        // then take the max over features to get the scaling constant ε.
        $globalMean = new Tensor([$d]);   // mean of each feature over all samples
        for ($i = 0; $i < $n; $i++) {
            $off = $i * $d;
            for ($j = 0; $j < $d; $j++) {
                $globalMean->buffer[$j] =
                    (float) $globalMean->buffer[$j] + (float) $X->buffer[$off + $j];
            }
        }
        for ($j = 0; $j < $d; $j++) {
            $globalMean->buffer[$j] = (float) $globalMean->buffer[$j] / $n;
        }

        $maxGlobalVar = 0.0;
        $globalVarAccum = array_fill(0, $d, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $off = $i * $d;
            for ($j = 0; $j < $d; $j++) {
                $diff              = (float) $X->buffer[$off + $j] - (float) $globalMean->buffer[$j];
                $globalVarAccum[$j] += $diff * $diff;
            }
        }
        for ($j = 0; $j < $d; $j++) {
            $gv = $globalVarAccum[$j] / $n;
            if ($gv > $maxGlobalVar) {
                $maxGlobalVar = $gv;
            }
        }

        // ε = var_smoothing × max_global_var
        // Add ε to every class-feature variance entry to prevent zero variance.
        $epsilon = $this->var_smoothing * $maxGlobalVar;

        if ($epsilon > 0.0) {
            for ($c = 0; $c < $nC; $c++) {
                $base = $c * $d;
                for ($j = 0; $j < $d; $j++) {
                    $var->buffer[$base + $j] = (float) $var->buffer[$base + $j] + $epsilon;
                }
            }
        }

        // ── Prior probabilities P(c) = n_c / n ────────────────────────────
        for ($c = 0; $c < $nC; $c++) {
            $classPrior->buffer[$c] = (float) $classCount->buffer[$c] / $n;
        }

        // ── Store fitted attributes ────────────────────────────────────────
        $this->theta_        = $theta;
        $this->var_          = $var;
        $this->class_prior_  = $classPrior;
        $this->class_count_  = $classCount;
        $this->classes_      = $classes;
        $this->n_classes_    = $nC;
        $this->n_features_in_ = $d;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels: argmax of log-posterior.
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Predicted labels [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        $logPost = $this->computeLogJoint($X);   // [n_samples, n_classes] PHP float[][]
        $m       = $X->shape[0];
        $nC      = $this->n_classes_;
        $out     = new Tensor([$m]);

        for ($i = 0; $i < $m; $i++) {
            $row     = $logPost[$i];
            $bestPos = 0;
            $bestVal = $row[0];
            for ($c = 1; $c < $nC; $c++) {
                if ($row[$c] > $bestVal) {
                    $bestVal = $row[$c];
                    $bestPos = $c;
                }
            }
            $out->buffer[$i] = (float) $this->classes_[$bestPos];
        }

        return $out;
    }

    /**
     * Predict class probability distributions via log-sum-exp normalisation.
     *
     * Returns a flat [n_samples, n_classes] Float32 Tensor (row-major):
     *   out[i * n_classes + c] = P(class c | X[i])
     *
     * Algorithm:
     *   logJoint[c] = log P(c) + Σ_j logLik_{c,j}    (computed in log-space)
     *   maxLog      = max_c logJoint[c]               (numeric stability anchor)
     *   log_denom   = maxLog + log(Σ_c exp(logJoint[c] − maxLog))
     *   P(c | x)    = exp(logJoint[c] − log_denom)   (guaranteed non-negative)
     *
     * @param Tensor $X  Feature matrix [n_samples, n_features]
     * @return Tensor    Probability matrix [n_samples, n_classes] (flat 1D)
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "GaussianNB::predict_proba() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $logPost = $this->computeLogJoint($X);
        $m       = $X->shape[0];
        $nC      = $this->n_classes_;
        $out     = new Tensor([$m, $nC]);

        for ($i = 0; $i < $m; $i++) {
            $row = $logPost[$i];

            // ── Log-sum-exp for numerical stability ────────────────────────
            //
            // Without the max-anchor, exp(logJoint[c]) may overflow (→ INF)
            // or underflow (→ 0).  Subtracting maxLog before exp keeps values
            // in a numerically safe range, then we add maxLog back via the
            // identity: log(Σ exp(a_c)) = maxLog + log(Σ exp(a_c − maxLog)).
            $maxLog = max($row);
            $sumExp = 0.0;
            for ($c = 0; $c < $nC; $c++) {
                $sumExp += exp($row[$c] - $maxLog);
            }
            $logDenom = $maxLog + log($sumExp);

            $base = $i * $nC;
            for ($c = 0; $c < $nC; $c++) {
                // exp(logJoint[c] - logDenom) = P(c | x)
                $out->buffer[$base + $c] = (float) exp($row[$c] - $logDenom);
            }
        }

        return $out;
    }

    /**
     * Accuracy score on test data.  Mirrors sklearn's ClassifierMixin.score().
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

    // ── Internal ───────────────────────────────────────────────────────────

    /**
     * Compute log-joint probabilities: logJoint[i][c] = log P(c) + Σ_j log N(x_{i,j}; θ_{c,j}, σ²_{c,j})
     *
     * The Gaussian log-PDF for one feature:
     *
     *   log N(x; μ, σ²) = −0.5 · log(2π σ²) − (x − μ)² / (2 σ²)
     *                    = −0.5 · [log(2π) + log(σ²)] − (x − μ)² / (2 σ²)
     *
     * The constant −0.5 · log(2π) per feature can be precomputed once.
     * Here it is absorbed into the per-class-feature term for clarity.
     *
     * Returns a PHP float[][] of shape [n_samples][n_classes].
     * Returning native PHP arrays avoids allocating a temporary Tensor and
     * makes the subsequent argmax / softmax steps direct PHP arithmetic.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return float[][]  [n_samples][n_classes]
     */
    private function computeLogJoint(Tensor $X): array
    {
        [$m, $d] = $X->shape;
        $nC      = $this->n_classes_;

        // Precompute log(P(c)) for each class
        $logPrior = [];
        for ($c = 0; $c < $nC; $c++) {
            $p = (float) $this->class_prior_->buffer[$c];
            // Guard: prior should always be > 0 (class seen in training), but clamp for safety
            $logPrior[$c] = $p > 0.0 ? log($p) : -1e38;
        }

        // Precompute per-class-feature constants to avoid recomputing inside the i-loop:
        //   logConst_{c,j} = −0.5 · log(2π · σ²_{c,j})
        // These are the same for every test sample; computing them once saves m multiplications.
        $LOG_2PI    = log(2.0 * M_PI);
        $logConst   = [];   // [nC][d]
        $twoVar     = [];   // [nC][d]  = 2 · σ²_{c,j}   (denominator in the exponent term)
        for ($c = 0; $c < $nC; $c++) {
            $base = $c * $d;
            for ($j = 0; $j < $d; $j++) {
                $v              = (float) $this->var_->buffer[$base + $j];
                $logConst[$c][$j] = -0.5 * ($LOG_2PI + log($v));
                $twoVar[$c][$j]   = 2.0 * $v;
            }
        }

        // ── Main loop: compute logJoint[i][c] for all i, c ────────────────
        $result = [];
        for ($i = 0; $i < $m; $i++) {
            $xOff  = $i * $d;
            $row   = [];

            for ($c = 0; $c < $nC; $c++) {
                $base    = $c * $d;
                $logJoint = $logPrior[$c];

                for ($j = 0; $j < $d; $j++) {
                    // log N(x_j; θ_{c,j}, σ²_{c,j})
                    //   = −0.5·log(2π σ²) − (x_j − θ)² / (2σ²)
                    $diff      = (float) $X->buffer[$xOff + $j] - (float) $this->theta_->buffer[$base + $j];
                    $logJoint += $logConst[$c][$j] - ($diff * $diff) / $twoVar[$c][$j];
                }

                $row[$c] = $logJoint;
            }

            $result[$i] = $row;
        }

        return $result;
    }

    private function checkFitted(): void
    {
        if (!isset($this->theta_)) {
            throw new \RuntimeException('GaussianNB is not fitted. Call fit() first.');
        }
    }
}
