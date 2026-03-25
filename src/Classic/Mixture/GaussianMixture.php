<?php

declare(strict_types=1);

namespace Pml\Classic\Mixture;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};
use Pml\Classic\Cluster\KMeans;

// ═══════════════════════════════════════════════════════════════════════════
//  GaussianMixture — sklearn.mixture.GaussianMixture
//
//  Gaussian Mixture Model (GMM) fitted via the Expectation-Maximisation (EM)
//  algorithm.  Supports full per-component covariance matrices.
//
//  ── Model ────────────────────────────────────────────────────────────────
//
//  The generative model for n_components mixture components:
//
//    p(x) = Σ_j  w_j · N(x | μ_j, Σ_j)
//
//  where:
//    w_j  — mixture weight (w_j > 0, Σ_j w_j = 1)
//    μ_j  — component mean  [d]
//    Σ_j  — component covariance  [d, d]  (positive definite)
//
//  ── EM Algorithm ─────────────────────────────────────────────────────────
//
//  E-step — compute soft responsibilities:
//
//    log π[i,j] = log w_j + log N(x_i | μ_j, Σ_j)
//
//    log N(x | μ, Σ) = − d/2·log(2π) − ½·log|Σ| − ½·(x−μ)^T Σ^{−1} (x−μ)
//
//  The Mahalanobis term (x−μ)^T Σ^{−1}(x−μ) is computed via the lower
//  Cholesky factor L of Σ:
//
//    Σ = L L^T   →   Σ^{−1} = (L^T)^{−1} L^{−1}
//
//    (x−μ)^T Σ^{−1}(x−μ) = ‖L^{−1}(x−μ)‖²
//
//  Forward-substituting L y = (x−μ) costs O(d²) per (sample, component).
//
//  Log-Sum-Exp trick prevents underflow when normalising responsibilities:
//
//    log Σ_j exp(log π[i,j]) = max_j + log Σ_j exp(log π[i,j] − max_j)
//
//    resp[i,j] = exp(log π[i,j] − logsumexp_j log π[i,:])
//
//  M-step — update parameters using weighted sufficient statistics:
//
//    N_j  = Σ_i resp[i,j]
//    w_j  = N_j / n
//    μ_j  = (1/N_j) · X^T @ resp[:,j]          ← single BLAS sgemv
//    Σ_j  = (1/N_j) · Σ_i resp[i,j]·δ_i·δ_i^T  + reg_covar·I
//           where δ_i = x_i − μ_j               ← n BLAS sger calls
//
//  Lower bound (ELBO) per sample:
//
//    L = (1/n) · Σ_i logsumexp_j log π[i,j]
//
//  Convergence: |L_new − L_old| < tol
//
//  ── Initialisation ───────────────────────────────────────────────────────
//
//  'kmeans'  — Run KMeans(k-means++, n_init=1) on X.  Set μ_j from cluster
//              centroids, Σ_j from per-cluster empirical covariance +
//              reg_covar·I, and w_j from cluster occupancy.
//
//  'random'  — Assign responsibilities from a random Dirichlet draw, then
//              run one M-step to derive initial parameters.
// ═══════════════════════════════════════════════════════════════════════════

final class GaussianMixture implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** Component means, shape [n_components, n_features]. */
    public readonly Tensor $means_;

    /**
     * Component covariance matrices.
     * Indexed by component j; each is a flat float[d*d] PHP array (row-major).
     * @var float[][]
     */
    public readonly array $covariances_;

    /** Mixture weights, shape [n_components].  Sums to 1. */
    public readonly Tensor $weights_;

    /** True if EM converged within max_iter. */
    public readonly bool $converged_;

    /** Number of EM iterations executed. */
    public readonly int $n_iter_;

    /** Final log-likelihood lower bound (per sample). */
    public readonly float $lower_bound_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int    $n_components  Number of mixture components k.
     * @param int    $max_iter      Maximum EM iterations.
     * @param int    $n_init        Number of independent EM runs; best
     *                              lower-bound is kept.
     * @param float  $tol           Convergence threshold on ELBO change.
     * @param float  $reg_covar     Regularisation added to each diagonal of Σ_j
     *                             to ensure positive definiteness.
     * @param int|null $random_state PHP mt_srand() seed.  null = no seeding.
     * @param string $init_params   'kmeans' or 'random'.
     */
    public function __construct(
        private readonly int    $n_components  = 1,
        private readonly int    $max_iter      = 100,
        private readonly int    $n_init        = 1,
        private readonly float  $tol           = 1e-3,
        private readonly float  $reg_covar     = 1e-6,
        private readonly ?int   $random_state  = null,
        private readonly string $init_params   = 'kmeans',
    ) {
        if ($n_components < 1) {
            throw new \InvalidArgumentException('GaussianMixture: n_components must be ≥ 1.');
        }
        if (!in_array($init_params, ['kmeans', 'random'], true)) {
            throw new \InvalidArgumentException(
                "GaussianMixture: init_params must be 'kmeans' or 'random'."
            );
        }
    }

    // ── Estimator ─────────────────────────────────────────────────────────

    /**
     * Fit the GMM via EM.  Runs n_init independent restarts.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Ignored (unsupervised).
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('GaussianMixture::fit() requires a 2D tensor.');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;
        $k                    = $this->n_components;

        if ($n < $k) {
            throw new \InvalidArgumentException(
                "GaussianMixture: n_samples={$n} < n_components={$k}."
            );
        }

        if ($this->random_state !== null) {
            mt_srand($this->random_state);
        }

        // Convert X to a 2D PHP float array once — avoids repeated FFI casts
        // in the inner EM loops (Cholesky, sger residual, etc.).
        $X2d = $this->tensorTo2dArray($X, $n, $d);

        $bestLowerBound = -INF;
        $bestMeans      = null;
        $bestCovs       = null;
        $bestWeights    = null;
        $bestConverged  = false;
        $bestNIter      = 0;

        for ($run = 0; $run < $this->n_init; $run++) {
            [$means, $covs, $weights] = $this->initialize($X, $X2d, $n, $d, $k);

            $lowerBound = -INF;
            $converged  = false;
            $iter       = 0;

            for ($iter = 1; $iter <= $this->max_iter; $iter++) {
                // ── E-step ────────────────────────────────────────────────
                [$resp, $newLowerBound] = $this->eStep($X2d, $means, $covs, $weights, $n, $d, $k);

                // ── Convergence check ─────────────────────────────────────
                if (abs($newLowerBound - $lowerBound) < $this->tol) {
                    $converged  = true;
                    $lowerBound = $newLowerBound;
                    break;
                }
                $lowerBound = $newLowerBound;

                // ── M-step ────────────────────────────────────────────────
                [$means, $covs, $weights] = $this->mStep($X, $X2d, $resp, $n, $d, $k);
            }

            if ($lowerBound > $bestLowerBound) {
                $bestLowerBound = $lowerBound;
                $bestMeans      = $means;
                $bestCovs       = $covs;
                $bestWeights    = $weights;
                $bestConverged  = $converged;
                $bestNIter      = $iter;
            }
        }

        // ── Store fitted state ────────────────────────────────────────────
        $meansTensor = new Tensor([$k, $d]);
        for ($j = 0; $j < $k; $j++) {
            for ($l = 0; $l < $d; $l++) {
                $meansTensor->buffer[$j * $d + $l] = (float) $bestMeans[$j][$l];
            }
        }

        $weightsTensor = new Tensor([$k]);
        for ($j = 0; $j < $k; $j++) {
            $weightsTensor->buffer[$j] = (float) $bestWeights[$j];
        }

        $this->means_       = $meansTensor;
        $this->covariances_ = $bestCovs;
        $this->weights_     = $weightsTensor;
        $this->converged_   = $bestConverged;
        $this->n_iter_      = $bestNIter;
        $this->lower_bound_ = $bestLowerBound;

        return $this;
    }

    // ── Predictor ─────────────────────────────────────────────────────────

    /**
     * Predict the component label (argmax responsibility) for each sample.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples] int labels in [0, n_components)
     */
    public function predict(Tensor $X): Tensor
    {
        $proba  = $this->predict_proba($X);
        $n      = $X->shape[0];
        $k      = $this->n_components;
        $labels = new Tensor([$n]);

        for ($i = 0; $i < $n; $i++) {
            $maxVal = -INF;
            $maxJ   = 0;
            for ($j = 0; $j < $k; $j++) {
                $v = (float) $proba->buffer[$i * $k + $j];
                if ($v > $maxVal) { $maxVal = $v; $maxJ = $j; }
            }
            $labels->buffer[$i] = (float) $maxJ;
        }
        return $labels;
    }

    /**
     * Posterior responsibilities — P(component j | x_i).
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples, n_components] row sums = 1
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('GaussianMixture::predict_proba() requires a 2D tensor.');
        }

        [$n, $d] = $X->shape;
        $k       = $this->n_components;
        $X2d     = $this->tensorTo2dArray($X, $n, $d);

        // Reconstruct PHP arrays from fitted Tensor state.
        $means   = $this->meansToPhpArray($k, $d);
        $weights = $this->weightsToPhpArray($k);
        $covs    = $this->covariances_;  // already PHP array

        [$resp]  = $this->eStep($X2d, $means, $covs, $weights, $n, $d, $k);

        $out = new Tensor([$n, $k]);
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $k; $j++) {
                $out->buffer[$i * $k + $j] = (float) $resp[$i][$j];
            }
        }
        return $out;
    }

    /**
     * Average log-likelihood of $X under the fitted model.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return float     Mean log p(x) per sample.
     */
    public function score(Tensor $X): float
    {
        $this->checkFitted();

        [$n, $d] = $X->shape;
        $k       = $this->n_components;
        $X2d     = $this->tensorTo2dArray($X, $n, $d);
        $means   = $this->meansToPhpArray($k, $d);
        $weights = $this->weightsToPhpArray($k);

        [, $lb] = $this->eStep($X2d, $means, $this->covariances_, $weights, $n, $d, $k);
        return $lb;
    }

    // ── EM steps ──────────────────────────────────────────────────────────

    /**
     * E-step: compute soft responsibilities and the log-likelihood lower bound.
     *
     * For each component j:
     *   1. Cholesky-decompose Σ_j → L_j.
     *   2. Compute log|Σ_j| = 2 · Σ_l log L_j[l,l].
     *   3. For each sample i:
     *      a. diff = x_i − μ_j
     *      b. Solve L_j y = diff  (forward substitution)
     *      c. log π[i,j] = log w_j − d/2·log(2π) − ½ log|Σ_j| − ½ ‖y‖²
     * 4. Normalise via Log-Sum-Exp → resp[i,j].
     *
     * @return array{0: float[][], 1: float}  [resp[n][k], mean_log_likelihood]
     */
    private function eStep(
        array $X2d,
        array $means,
        array $covs,
        array $weights,
        int $n,
        int $d,
        int $k
    ): array {
        $logTwoPi = log(2.0 * M_PI);

        // Pre-compute Cholesky and log-determinant per component.
        $cholsL  = [];
        $logDets = [];
        for ($j = 0; $j < $k; $j++) {
            $L           = self::cholesky($covs[$j], $this->reg_covar, $d);
            $cholsL[$j]  = $L;
            $logDets[$j] = self::logDetFromChol($L, $d);
        }

        // Compute log π[i,j] for all i, j.
        $logProb = [];
        for ($i = 0; $i < $n; $i++) {
            $logProb[$i] = [];
            for ($j = 0; $j < $k; $j++) {
                // diff = x_i − μ_j
                $diff = [];
                for ($l = 0; $l < $d; $l++) {
                    $diff[$l] = $X2d[$i][$l] - $means[$j][$l];
                }
                $mah = self::mahalanobisSq($diff, $cholsL[$j], $d);
                $logProb[$i][$j] = log(max(1e-300, $weights[$j]))
                                   - 0.5 * ($d * $logTwoPi + $logDets[$j] + $mah);
            }
        }

        // Normalise via Log-Sum-Exp; accumulate lower bound.
        $resp       = [];
        $totalLogLk = 0.0;

        for ($i = 0; $i < $n; $i++) {
            $lse          = self::logSumExp($logProb[$i]);
            $totalLogLk  += $lse;
            $resp[$i]     = [];
            $rowSum       = 0.0;
            for ($j = 0; $j < $k; $j++) {
                $r            = exp($logProb[$i][$j] - $lse);
                $resp[$i][$j] = $r;
                $rowSum      += $r;
            }
            // Re-normalise to guard against exp rounding.
            if ($rowSum > 1e-12) {
                for ($j = 0; $j < $k; $j++) {
                    $resp[$i][$j] /= $rowSum;
                }
            }
        }

        return [$resp, $totalLogLk / $n];
    }

    /**
     * M-step: update means, covariances, and weights from responsibilities.
     *
     * μ_j  update: single sgemv call — X^T @ resp[:,j] / N_j
     * Σ_j  update: n sger calls (BLAS rank-1) — Σ_i resp[i,j]/N_j · δ_i δ_i^T
     *
     * @return array{0: float[][], 1: float[][], 2: float[]}
     *   [means[k][d], covariances[k][d*d], weights[k]]
     */
    private function mStep(
        Tensor $X,
        array $X2d,
        array $resp,
        int $n,
        int $d,
        int $k
    ): array {
        $blas = BlasEngine::get()->ffi;

        // ── Effective cluster sizes N_j = Σ_i resp[i,j] ──────────────────
        $Nj      = array_fill(0, $k, 0.0);
        $respFlat = [];  // resp[:,j] as a contiguous Tensor per component

        for ($j = 0; $j < $k; $j++) {
            $respTensor = new Tensor([$n]);
            for ($i = 0; $i < $n; $i++) {
                $v                     = $resp[$i][$j];
                $Nj[$j]               += $v;
                $respTensor->buffer[$i] = (float) $v;
            }
            $respFlat[$j] = $respTensor;
        }

        $means   = [];
        $covs    = [];
        $weights = [];

        for ($j = 0; $j < $k; $j++) {
            $nj = max($Nj[$j], 1e-10); // prevent division by zero

            // ── Mean update: μ_j = X^T @ resp[:,j] / N_j via sgemv ──────────
            //
            // sgemv(RowMajor, Trans, n, d, 1/nj, X[n,d], d, resp_j[n], 1, 0, mu[d], 1)
            // Computes: μ[l] = (1/nj) · Σ_i resp[i,j] · X[i,l]
            $muTensor = new Tensor([$d]);
            $blas->cblas_sgemv(
                101,                // CblasRowMajor
                112,                // CblasTrans — X^T
                $n, $d,
                1.0 / $nj,
                $X->buffer, $d,
                $respFlat[$j]->buffer, 1,
                0.0,
                $muTensor->buffer, 1
            );

            $mu = [];
            for ($l = 0; $l < $d; $l++) {
                $mu[$l] = (float) $muTensor->buffer[$l];
            }
            $means[$j] = $mu;

            // ── Covariance update: Σ_j = Σ_i r_ij/N_j · (x_i−μ)(x_i−μ)^T ─
            //
            // Using n BLAS sger rank-1 updates on a [d×d] Tensor.
            // sger(RowMajor, d, d, r_ij/N_j, diff[d], 1, diff[d], 1, Sigma[d,d], d)
            // Adds: Σ_j += α · diff · diff^T
            $sigmaTensor = new Tensor([$d, $d]);
            $diffBuf     = new Tensor([$d]);

            for ($i = 0; $i < $n; $i++) {
                $alpha = (float) $resp[$i][$j] / $nj;
                for ($l = 0; $l < $d; $l++) {
                    $diffBuf->buffer[$l] = (float) ($X2d[$i][$l] - $mu[$l]);
                }
                $blas->cblas_sger(
                    101,                       // CblasRowMajor
                    $d, $d,
                    $alpha,
                    $diffBuf->buffer, 1,
                    $diffBuf->buffer, 1,
                    $sigmaTensor->buffer, $d
                );
            }

            // Add reg_covar to diagonal to ensure positive definiteness.
            $flatSigma = [];
            for ($r = 0; $r < $d; $r++) {
                for ($c = 0; $c < $d; $c++) {
                    $val = (float) $sigmaTensor->buffer[$r * $d + $c];
                    if ($r === $c) { $val += $this->reg_covar; }
                    $flatSigma[$r * $d + $c] = $val;
                }
            }
            $covs[$j] = $flatSigma;

            // ── Weight update: w_j = N_j / n ─────────────────────────────────
            $weights[$j] = $nj / $n;
        }

        return [$means, $covs, $weights];
    }

    // ── Initialisation ────────────────────────────────────────────────────

    /**
     * @return array{0: float[][], 1: float[][], 2: float[]}
     *   [means[k][d], covariances[k][d*d], weights[k]]
     */
    private function initialize(
        Tensor $X,
        array $X2d,
        int $n,
        int $d,
        int $k
    ): array {
        if ($this->init_params === 'kmeans') {
            return $this->initKMeans($X, $X2d, $n, $d, $k);
        }
        return $this->initRandom($X2d, $n, $d, $k);
    }

    /**
     * 'kmeans' initialisation: run KMeans(k-means++, n_init=1) to seed means.
     * Per-cluster empirical covariance + reg_covar·I provides warm covariances.
     * Weights from cluster occupancy fractions.
     */
    private function initKMeans(Tensor $X, array $X2d, int $n, int $d, int $k): array
    {
        $km = new KMeans(
            n_clusters:   $k,
            max_iter:     100,
            n_init:       1,
            init:         'k-means++',
        );
        $km->fit($X);

        $means   = [];
        $covs    = [];
        $weights = [];

        // Collect cluster assignments.
        $clusters = array_fill(0, $k, []);
        for ($i = 0; $i < $n; $i++) {
            $label        = (int) $km->labels_->buffer[$i];
            $clusters[$label][] = $i;
        }

        for ($j = 0; $j < $k; $j++) {
            // Mean from KMeans centroid.
            $mu = [];
            for ($l = 0; $l < $d; $l++) {
                $mu[$l] = (float) $km->cluster_centers_->buffer[$j * $d + $l];
            }
            $means[$j] = $mu;

            // Empirical covariance of cluster j + reg diagonal.
            $flatSigma = array_fill(0, $d * $d, 0.0);
            $clusterN  = count($clusters[$j]);

            if ($clusterN > 1) {
                foreach ($clusters[$j] as $idx) {
                    for ($r = 0; $r < $d; $r++) {
                        $dr = $X2d[$idx][$r] - $mu[$r];
                        for ($c = 0; $c < $d; $c++) {
                            $flatSigma[$r * $d + $c] += $dr * ($X2d[$idx][$c] - $mu[$c]);
                        }
                    }
                }
                for ($r = 0; $r < $d * $d; $r++) {
                    $flatSigma[$r] /= $clusterN;
                }
            }
            // Add reg_covar to diagonal.
            for ($r = 0; $r < $d; $r++) {
                $flatSigma[$r * $d + $r] += $this->reg_covar;
            }
            $covs[$j] = $flatSigma;

            $weights[$j] = max($clusterN, 1) / $n;
        }

        // Renormalise weights.
        $wSum = array_sum($weights);
        for ($j = 0; $j < $k; $j++) {
            $weights[$j] /= $wSum;
        }

        return [$means, $covs, $weights];
    }

    /**
     * 'random' initialisation: random Dirichlet-like responsibility draw,
     * then one M-step to derive initial parameters.
     */
    private function initRandom(array $X2d, int $n, int $d, int $k): array
    {
        // Draw uniform responsibilities and row-normalise.
        $resp = [];
        for ($i = 0; $i < $n; $i++) {
            $row    = [];
            $rowSum = 0.0;
            for ($j = 0; $j < $k; $j++) {
                $v      = (float) mt_rand() / (float) mt_getrandmax() + 1e-10;
                $row[]  = $v;
                $rowSum += $v;
            }
            for ($j = 0; $j < $k; $j++) {
                $row[$j] /= $rowSum;
            }
            $resp[$i] = $row;
        }

        // Build a minimal Tensor for X to pass to mStep.
        // Re-use the flat Tensor that the caller has; we need a Tensor $X here.
        // Since initRandom doesn't receive $X, we rebuild it from $X2d.
        $xTmp = new Tensor([$n, $d]);
        for ($i = 0; $i < $n; $i++) {
            for ($l = 0; $l < $d; $l++) {
                $xTmp->buffer[$i * $d + $l] = (float) $X2d[$i][$l];
            }
        }

        return $this->mStep($xTmp, $X2d, $resp, $n, $d, $k);
    }

    // ── Numerical utilities ───────────────────────────────────────────────

    /**
     * Cholesky decomposition of a symmetric positive-definite matrix A[d,d].
     *
     * Returns the lower triangular factor L such that A = L L^T.
     *
     * Algorithm: Cholesky-Banachiewicz (row-wise):
     *   L[i,j] = (A[i,j] − Σ_{k<j} L[i,k]·L[j,k]) / L[j,j]   for j < i
     *   L[i,i] = sqrt(A[i,i] − Σ_{k<i} L[i,k]²)
     *
     * The reg_covar regularisation in the M-step ensures A is sufficiently
     * positive definite to avoid negative radicands.
     *
     * @param float[] $A       Flat row-major [d*d] covariance matrix.
     * @param float   $reg     Extra ridge added to the diagonal before factoring.
     * @param int     $d       Dimension.
     * @return float[]         Flat row-major [d*d] lower triangular matrix L.
     */
    private static function cholesky(array $A, float $reg, int $d): array
    {
        $L = array_fill(0, $d * $d, 0.0);

        for ($i = 0; $i < $d; $i++) {
            for ($j = 0; $j <= $i; $j++) {
                $sum = ($i === $j)
                    ? $A[$i * $d + $i] + $reg  // diagonal: add regularisation
                    : $A[$i * $d + $j];

                for ($m = 0; $m < $j; $m++) {
                    $sum -= $L[$i * $d + $m] * $L[$j * $d + $m];
                }

                if ($i === $j) {
                    // Diagonal: sqrt; clamp to avoid sqrt of negative due to float noise.
                    $L[$i * $d + $i] = sqrt(max($sum, 1e-14));
                } else {
                    // Off-diagonal: divide by diagonal element.
                    $L[$i * $d + $j] = ($L[$j * $d + $j] > 1e-14)
                        ? $sum / $L[$j * $d + $j]
                        : 0.0;
                }
            }
        }
        return $L;
    }

    /**
     * log|Σ| from Cholesky factor L where Σ = L L^T.
     *
     * log|L L^T| = 2 · log|L| = 2 · Σ_i log L[i,i]
     *
     * (|L| = product of diagonal entries since L is triangular.)
     *
     * @param float[] $L  Flat row-major [d*d] lower triangular Cholesky factor.
     */
    private static function logDetFromChol(array $L, int $d): float
    {
        $logDet = 0.0;
        for ($i = 0; $i < $d; $i++) {
            $logDet += log(max($L[$i * $d + $i], 1e-14));
        }
        return 2.0 * $logDet;
    }

    /**
     * Mahalanobis distance squared: (x−μ)^T Σ^{−1} (x−μ) = ‖L^{−1} diff‖²
     *
     * Solved via forward substitution on L y = diff:
     *   y[i] = (diff[i] − Σ_{j<i} L[i,j] · y[j]) / L[i,i]
     *
     * Then ‖y‖² = Σ_i y[i]².
     *
     * @param float[] $diff  Difference vector x_i − μ_j  (length d).
     * @param float[] $L     Flat row-major [d*d] lower triangular Cholesky factor.
     */
    private static function mahalanobisSq(array $diff, array $L, int $d): float
    {
        $y = array_fill(0, $d, 0.0);
        for ($i = 0; $i < $d; $i++) {
            $s = $diff[$i];
            for ($j = 0; $j < $i; $j++) {
                $s -= $L[$i * $d + $j] * $y[$j];
            }
            $y[$i] = ($L[$i * $d + $i] > 1e-14) ? $s / $L[$i * $d + $i] : 0.0;
        }

        $sq = 0.0;
        foreach ($y as $v) { $sq += $v * $v; }
        return $sq;
    }

    /**
     * Numerically stable log Σ exp(a_j) via the Log-Sum-Exp identity:
     *
     *   log Σ_j exp(a_j) = max_j(a_j) + log Σ_j exp(a_j − max_j(a_j))
     *
     * Subtracting the maximum prevents overflow/underflow in the inner exp.
     *
     * @param float[] $vals  Input log-probabilities (one per component).
     */
    private static function logSumExp(array $vals): float
    {
        $max = max($vals);
        if ($max === -INF) { return -INF; }

        $sum = 0.0;
        foreach ($vals as $v) {
            $sum += exp($v - $max);
        }
        return $max + log(max($sum, 1e-300));
    }

    // ── Tensor ↔ PHP array helpers ────────────────────────────────────────

    /**
     * Convert a 2D Tensor [n, d] to a PHP float[][] array.
     * Done once per fit/predict call to avoid repeated FFI buffer casts.
     */
    private function tensorTo2dArray(Tensor $X, int $n, int $d): array
    {
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($l = 0; $l < $d; $l++) {
                $row[$l] = (float) $X->buffer[$i * $d + $l];
            }
            $out[$i] = $row;
        }
        return $out;
    }

    /**
     * Extract means from the fitted Tensor into a PHP float[][] [k][d] array.
     */
    private function meansToPhpArray(int $k, int $d): array
    {
        $out = [];
        for ($j = 0; $j < $k; $j++) {
            $row = [];
            for ($l = 0; $l < $d; $l++) {
                $row[$l] = (float) $this->means_->buffer[$j * $d + $l];
            }
            $out[$j] = $row;
        }
        return $out;
    }

    /** Extract weights from the fitted Tensor into a PHP float[] array. */
    private function weightsToPhpArray(int $k): array
    {
        $out = [];
        for ($j = 0; $j < $k; $j++) {
            $out[$j] = (float) $this->weights_->buffer[$j];
        }
        return $out;
    }

    private function checkFitted(): void
    {
        if (!isset($this->means_)) {
            throw new \RuntimeException('GaussianMixture is not fitted. Call fit() first.');
        }
    }
}
