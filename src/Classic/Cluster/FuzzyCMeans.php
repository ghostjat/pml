<?php

declare(strict_types=1);

namespace Pml\Classic\Cluster;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  FuzzyCMeans — Fuzzy C-Means soft clustering
//               (Bezdek, 1981; Dunn, 1973)
//
//  Unlike hard K-Means, each point x_i belongs to ALL k clusters with a
//  degree of membership u_ij ∈ (0, 1).  Constraint: Σ_j u_ij = 1 for all i.
//
//  ── Model ────────────────────────────────────────────────────────────────
//
//  Minimise the fuzzy objective:
//
//    J_m = Σ_i Σ_j  u_ij^m  ·  ‖x_i − v_j‖²
//
//  where:
//    u_ij  — membership of point i in cluster j
//    v_j   — cluster centroid
//    m     — fuzzifier (m > 1; m=2 is standard; m→1 recovers hard K-Means;
//             m→∞ gives uniform memberships)
//
//  ── Update Rules ─────────────────────────────────────────────────────────
//
//  Centroid update (M-step):
//
//    v_j = (Σ_i u_ij^m · x_i) / (Σ_i u_ij^m)
//
//    Computed via two BLAS calls per cluster:
//      1. w_j[n]  = u[:,j]^m  (element-wise power, PHP loop)
//      2. v_j[d]  = X^T @ w_j / Σ w_j  (sgemv: single BLAS-2 call)
//
//  Membership update (E-step):
//
//    u_ij = 1 / Σ_l (d_ij / d_il)^(2/(m−1))
//
//    where d_ij = ‖x_i − v_j‖²  (squared Euclidean distance).
//
//    Special cases:
//      • d_ij = 0 → u_ij = 1, u_il = 0 for l ≠ j  (point exactly at centroid j)
//      • 2/(m−1) = 1 when m = 3, 2 when m = 2, etc.
//
//  The distances D[n,k] are computed each iteration via the BLAS expansion:
//    D = xnorm ⊗ 1_k + 1_n ⊗ vnorm − 2 · X @ V^T
//
//  ── Convergence ──────────────────────────────────────────────────────────
//
//  Stop when the Frobenius norm of the membership change drops below tol:
//    ‖U_new − U_old‖_F < tol
//
//  ── Complexity per iteration ─────────────────────────────────────────────
//
//    Distance matrix:  O(n·k·d) via sgemm  (BLAS-3)
//    Centroid update:  O(n·k) PHP + O(k·n·d) via k×sgemv
//    Membership update: O(n·k²) PHP
//
//    Total per iter: O(n·k·d) BLAS + O(n·k²) PHP
// ═══════════════════════════════════════════════════════════════════════════

final class FuzzyCMeans implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** Cluster centroids, shape [n_clusters, n_features]. */
    public readonly Tensor $cluster_centers_;

    /**
     * Fuzzy membership matrix, shape [n_samples, n_clusters].
     * Entry [i,j] = degree of membership of sample i in cluster j.
     * Each row sums to 1.0.
     */
    public readonly Tensor $membership_;

    /** Hard cluster labels (argmax of membership), shape [n_samples]. */
    public readonly Tensor $labels_;

    /** Fuzzy objective value J_m at convergence. */
    public readonly float $inertia_;

    /** Number of iterations until convergence. */
    public readonly int $n_iter_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int      $n_clusters    Number of clusters k.
     * @param float    $m             Fuzzifier (m > 1).  m=2 is the standard choice.
     *                                Larger m → softer memberships (more uniform).
     *                                As m→1, recovers hard K-Means assignments.
     * @param int      $max_iter      Maximum iterations.
     * @param float    $tol           Convergence threshold on ‖U_new − U_old‖_F.
     * @param int|null $random_state  PHP mt_srand() seed.  null = no seeding.
     */
    public function __construct(
        private readonly int   $n_clusters   = 3,
        private readonly float $m            = 2.0,
        private readonly int   $max_iter     = 150,
        private readonly float $tol          = 1e-4,
        private readonly ?int  $random_state = null,
    ) {
        if ($n_clusters < 1) {
            throw new \InvalidArgumentException('FuzzyCMeans: n_clusters must be ≥ 1.');
        }
        if ($m <= 1.0) {
            throw new \InvalidArgumentException(
                "FuzzyCMeans: fuzzifier m must be > 1 (got {$m}). Use m=2 for the standard FCM."
            );
        }
    }

    // ── Estimator ─────────────────────────────────────────────────────────

    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        $this->fit_predict($X);
        return $this;
    }

    // ── Predictor ─────────────────────────────────────────────────────────

    /**
     * Hard cluster label (argmax of membership) for new samples.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples] int labels
     */
    public function predict(Tensor $X): Tensor
    {
        $proba  = $this->predict_proba($X);
        $n      = $X->shape[0];
        $k      = $this->n_clusters;
        $labels = new Tensor([$n]);

        for ($i = 0; $i < $n; $i++) {
            $maxVal = -1.0;
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
     * Soft memberships for new (unseen) samples.
     *
     * Uses the fitted centroids V: compute distances, then apply the
     * FCM membership formula without re-training.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples, n_clusters] row-stochastic membership matrix
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('FuzzyCMeans::predict_proba() requires a 2D tensor.');
        }

        [$n, $d] = $X->shape;
        $k       = $this->n_clusters;
        $D       = $this->distanceMatrix($X, $this->cluster_centers_, $n, $k, $d);
        $U       = $this->computeMemberships($D, $n, $k);

        $out = new Tensor([$n, $k]);
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $k; $j++) {
                $out->buffer[$i * $k + $j] = (float) $U[$i][$j];
            }
        }
        return $out;
    }

    // ── Core algorithm ────────────────────────────────────────────────────

    /**
     * Run Fuzzy C-Means and return the hard label Tensor.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Ignored (unsupervised).
     * @return Tensor         [n_samples] hard labels (argmax of membership)
     */
    public function fit_predict(Tensor $X, ?Tensor $y = null): Tensor
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'FuzzyCMeans: X must be 2-D [n_samples, n_features].'
            );
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;
        $k                    = $this->n_clusters;
        $blas                 = BlasEngine::get()->ffi;

        if ($this->random_state !== null) {
            mt_srand($this->random_state);
        }

        // ── Initialise membership matrix U[n,k] uniformly at random ─────────
        //
        // Each row is drawn uniformly from the (k−1)-simplex:
        //   sample k uniform deviates, divide by their sum.
        $U = $this->initMemberships($n, $k);

        // ── Allocate centroid Tensor V[k,d] ──────────────────────────────────
        $V       = new Tensor([$k, $d]);
        $iter    = 0;
        $inertia = INF;

        for ($iter = 1; $iter <= $this->max_iter; $iter++) {
            // ── M-step: update centroids v_j = (Σ u_ij^m · x_i) / Σ u_ij^m ──
            //
            // For each cluster j:
            //   w_j[i] = u_ij^m                           (PHP element-wise pow)
            //   v_j    = X^T @ w_j / Σ w_j               (single BLAS sgemv)
            \FFI::memset($V->buffer, 0, $k * $d * 4);

            for ($j = 0; $j < $k; $j++) {
                $wTensor = new Tensor([$n]);
                $wSum    = 0.0;
                for ($i = 0; $i < $n; $i++) {
                    $w                     = $U[$i][$j] ** $this->m;
                    $wTensor->buffer[$i]   = (float) $w;
                    $wSum                 += $w;
                }
                $wSum = max($wSum, 1e-12);

                // sgemv(Trans, n, d, 1/wSum, X, d, w, 1, 0, v_j, 1)
                // → v_j[l] = (1/wSum) · Σ_i w[i] · X[i,l]
                $vPtr = \FFI::cast('float*', \FFI::addr($V->buffer[$j * $d]));
                $blas->cblas_sgemv(
                    101,              // CblasRowMajor
                    112,              // CblasTrans — X^T
                    $n, $d,
                    1.0 / $wSum,
                    $X->buffer, $d,
                    $wTensor->buffer, 1,
                    0.0,
                    $vPtr, 1
                );
            }

            // ── Compute squared distance matrix D[n,k] ───────────────────────
            //   D[i,j] = ‖x_i − v_j‖²  via BLAS expansion (see distanceMatrix)
            $D = $this->distanceMatrix($X, $V, $n, $k, $d);

            // ── E-step: update fuzzy memberships ─────────────────────────────
            $Unew    = $this->computeMemberships($D, $n, $k);

            // ── Convergence: ‖U_new − U_old‖_F < tol ────────────────────────
            $frobSq = 0.0;
            for ($i = 0; $i < $n; $i++) {
                for ($j = 0; $j < $k; $j++) {
                    $diff    = $Unew[$i][$j] - $U[$i][$j];
                    $frobSq += $diff * $diff;
                }
            }

            $U = $Unew;
            if (sqrt($frobSq) < $this->tol) {
                break;
            }
        }

        // ── Compute fuzzy inertia J_m = Σ_i Σ_j u_ij^m · d_ij ──────────────
        $D = $this->distanceMatrix($X, $V, $n, $k, $d);
        $inertia = 0.0;
        for ($i = 0; $i < $n; $i++) {
            for ($j = 0; $j < $k; $j++) {
                $inertia += ($U[$i][$j] ** $this->m) * max(0.0, (float) $D->buffer[$i * $k + $j]);
            }
        }

        // ── Store fitted state ────────────────────────────────────────────────
        $this->cluster_centers_ = $V;

        $memberTensor = new Tensor([$n, $k]);
        $labelTensor  = new Tensor([$n]);

        for ($i = 0; $i < $n; $i++) {
            $maxVal = -1.0;
            $maxJ   = 0;
            for ($j = 0; $j < $k; $j++) {
                $u                                 = (float) $U[$i][$j];
                $memberTensor->buffer[$i * $k + $j] = $u;
                if ($u > $maxVal) { $maxVal = $u; $maxJ = $j; }
            }
            $labelTensor->buffer[$i] = (float) $maxJ;
        }

        $this->membership_ = $memberTensor;
        $this->labels_     = $labelTensor;
        $this->inertia_    = $inertia;
        $this->n_iter_     = $iter;

        return $labelTensor;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /**
     * FCM membership update formula:
     *
     *   u_ij = 1 / Σ_l (d_ij / d_il)^(2/(m−1))
     *
     * where d_ij = D[i,j] (squared Euclidean distance).
     *
     * Corner case: if d_ij = 0 for some j, the point x_i lies exactly on
     * centroid v_j.  Set u_ij = 1 and all other memberships to 0.
     *
     * @return float[][]  U[n][k], row-stochastic.
     */
    private function computeMemberships(Tensor $D, int $n, int $k): array
    {
        $exp = 2.0 / ($this->m - 1.0); // exponent in the membership formula
        $U   = [];

        for ($i = 0; $i < $n; $i++) {
            $row       = array_fill(0, $k, 0.0);
            $zeroClust = -1; // cluster at exactly zero distance (if any)

            for ($j = 0; $j < $k; $j++) {
                if ((float) $D->buffer[$i * $k + $j] <= 0.0) {
                    $zeroClust = $j;
                    break;
                }
            }

            if ($zeroClust >= 0) {
                // Point sits exactly on centroid $zeroClust — hard assignment.
                $row[$zeroClust] = 1.0;
            } else {
                // Standard formula: u_ij = 1 / Σ_l (d_ij/d_il)^exp
                for ($j = 0; $j < $k; $j++) {
                    $dij = (float) $D->buffer[$i * $k + $j];
                    $sum = 0.0;
                    for ($l = 0; $l < $k; $l++) {
                        $dil = (float) $D->buffer[$i * $k + $l];
                        $sum += ($dil > 0.0) ? ($dij / $dil) ** $exp : 0.0;
                    }
                    $row[$j] = ($sum > 1e-14) ? 1.0 / $sum : 0.0;
                }

                // Renormalise to enforce Σ_j u_ij = 1 exactly.
                $rowSum = array_sum($row);
                if ($rowSum > 1e-14) {
                    for ($j = 0; $j < $k; $j++) { $row[$j] /= $rowSum; }
                } else {
                    // Degenerate: assign uniform membership.
                    for ($j = 0; $j < $k; $j++) { $row[$j] = 1.0 / $k; }
                }
            }

            $U[$i] = $row;
        }
        return $U;
    }

    /**
     * Initialise U[n,k] with random row-stochastic values.
     * Each row is sampled uniformly from the (k−1)-simplex.
     *
     * @return float[][]
     */
    private function initMemberships(int $n, int $k): array
    {
        $U = [];
        for ($i = 0; $i < $n; $i++) {
            $row    = [];
            $rowSum = 0.0;
            for ($j = 0; $j < $k; $j++) {
                // Uniform(0,1) + epsilon avoids exact-zero memberships.
                $v      = ((float) mt_rand() / (float) mt_getrandmax()) + 1e-6;
                $row[]  = $v;
                $rowSum += $v;
            }
            for ($j = 0; $j < $k; $j++) { $row[$j] /= $rowSum; }
            $U[$i] = $row;
        }
        return $U;
    }

    /**
     * Squared Euclidean distance matrix D[n,k] via the BLAS expansion:
     *
     *   D[i,j] = ‖x_i − v_j‖²
     *          = ‖x_i‖² + ‖v_j‖² − 2·(x_i · v_j)
     *
     * Step-by-step:
     *   1. xnorm[i] = sdot(X[i], X[i])           O(n) sdot calls
     *   2. vnorm[j] = sdot(V[j], V[j])           O(k) sdot calls
     *   3. D = −2 · X @ V^T                       1 sgemm call
     *   4. D += xnorm ⊗ ones_k  (sger)
     *   5. D += ones_n ⊗ vnorm  (sger)
     */
    private function distanceMatrix(Tensor $X, Tensor $V, int $n, int $k, int $d): Tensor
    {
        $blas = BlasEngine::get()->ffi;

        $xnorm = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $xPtr              = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
            $xnorm->buffer[$i] = $blas->cblas_sdot($d, $xPtr, 1, $xPtr, 1);
        }

        $vnorm = new Tensor([$k]);
        for ($j = 0; $j < $k; $j++) {
            $vPtr              = \FFI::cast('float*', \FFI::addr($V->buffer[$j * $d]));
            $vnorm->buffer[$j] = $blas->cblas_sdot($d, $vPtr, 1, $vPtr, 1);
        }

        $D = new Tensor([$n, $k]);
        $blas->cblas_sgemm(101, 111, 112, $n, $k, $d, -2.0,
            $X->buffer, $d, $V->buffer, $d, 0.0, $D->buffer, $k);

        $ones_k = Tensor::ones([$k]);
        $blas->cblas_sger(101, $n, $k, 1.0, $xnorm->buffer, 1, $ones_k->buffer, 1, $D->buffer, $k);

        $ones_n = Tensor::ones([$n]);
        $blas->cblas_sger(101, $n, $k, 1.0, $ones_n->buffer, 1, $vnorm->buffer, 1, $D->buffer, $k);

        return $D;
    }

    private function checkFitted(): void
    {
        if (!isset($this->cluster_centers_)) {
            throw new \RuntimeException('FuzzyCMeans is not fitted. Call fit() first.');
        }
    }
}
