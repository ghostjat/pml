<?php

declare(strict_types=1);

namespace Pml\Classic\Cluster;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  MeanShift — sklearn.cluster.MeanShift
//
//  Non-parametric mode-seeking clustering algorithm.
//  (Comaniciu & Meer, 2002; Fukunaga & Hostetler, 1975)
//
//  ── Intuition ────────────────────────────────────────────────────────────
//
//  Mean Shift treats the dataset as a kernel density estimate p(x).
//  Starting from each data point, the algorithm iteratively moves the point
//  "uphill" along the density gradient until it converges to a mode (local
//  maximum).  Points that converge to the same mode are co-clustered.
//
//  Key advantages over K-Means:
//    • No need to specify the number of clusters k (it is discovered).
//    • Can detect arbitrary cluster shapes.
//    • Robust to outliers (low-density regions stay isolated).
//
//  ── Flat Kernel ──────────────────────────────────────────────────────────
//
//  For sample x (called a "seed"), the Mean Shift vector is:
//
//    m(x) = [Σ_{x_i ∈ N(x)} x_i] / |N(x)|  −  x
//
//  where N(x) = { x_i : ‖x − x_i‖ ≤ bandwidth }.
//
//  The flat (uniform) kernel is used.  The Gaussian kernel (exp-weighted)
//  is available via the $kernel parameter.
//
//  ── Batch BLAS Iteration ─────────────────────────────────────────────────
//
//  All n seeds are updated simultaneously each iteration:
//
//  1. D[n,n] = squared distance matrix (seeds vs X) via sgemm expansion.
//     (If seeds = X initially, this is the n×n self-distance matrix.)
//
//  2. Build weight matrix W[n,n]:
//       Flat kernel:     W[i,j] = 1  if D[i,j] ≤ bw²,  else 0
//       Gaussian kernel: W[i,j] = exp(−D[i,j] / (2·bw²))
//
//  3. Compute new seed positions:
//       numerator[i,:]   = Σ_j W[i,j] · X[j,:]   ← sgemm: W[n,n] @ X[n,d]
//       counts[i]        = Σ_j W[i,j]             ← sgemv or PHP sum
//       new_seed[i,:]    = numerator[i,:] / counts[i]
//
//  The sgemm in step 3 makes each iteration O(n²d) in BLAS — identical cost
//  to the distance matrix, but native C throughput.
//
//  ── Mode Merging ─────────────────────────────────────────────────────────
//
//  After convergence, modes (final seed positions) within distance
//  cluster_merge_tol = bandwidth / 2 of each other are merged greedily:
//  pick the first unmerged mode, absorb all modes within the merge radius,
//  and recurse.  The merged mode = mean of absorbed positions.
//
//  ── KDTree Note ──────────────────────────────────────────────────────────
//
//  When Pml\Classic\Spatial\KDTree becomes available, the inner loop of
//  neighbour lookup (step 2) can be replaced with KDTree::query_radius(),
//  reducing neighbour extraction from O(n²) to O(n log n) on average.
//
//  ── Memory ───────────────────────────────────────────────────────────────
//
//  The n×n weight matrix W is built as a Tensor buffer (float32):
//    n=1000 → 4 MB  ·  n=5000 → 100 MB  ·  n=10000 → 400 MB
//  The distance matrix D is freed immediately after building W.
// ═══════════════════════════════════════════════════════════════════════════

final class MeanShift implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** Cluster mode positions, shape [n_clusters_, n_features]. */
    public readonly Tensor $cluster_centers_;

    /** Cluster label for each training sample, shape [n_samples]. */
    public readonly Tensor $labels_;

    /** Number of discovered clusters. */
    public readonly int $n_clusters_;

    /** Number of Mean Shift iterations executed. */
    public readonly int $n_iter_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float  $bandwidth         Window radius.  All points within this
     *                                  distance of a seed contribute to its
     *                                  Mean Shift update.
     *                                  Rule of thumb: set to the typical
     *                                  inter-cluster distance.
     * @param string $kernel            Kernel type: 'flat' (uniform window) or
     *                                  'gaussian' (exp-weighted).
     * @param int    $max_iter          Maximum Mean Shift iterations.
     * @param float  $tol               Convergence threshold — max seed shift
     *                                  across all points (L∞ norm of shifts).
     * @param float  $cluster_merge_tol Seeds closer than this distance after
     *                                  convergence are merged into one cluster.
     *                                  Defaults to bandwidth / 2.
     * @param bool   $cluster_all       If true, assign every sample (including
     *                                  potential outliers) to the nearest
     *                                  discovered mode.  If false, samples
     *                                  farther than $bandwidth from any mode
     *                                  receive label −1 (noise).
     */
    public function __construct(
        private readonly float   $bandwidth,
        private readonly string  $kernel            = 'flat',
        private readonly int     $max_iter          = 300,
        private readonly float   $tol               = 1e-3,
        private readonly ?float  $cluster_merge_tol = null,
        private readonly bool    $cluster_all       = true,
    ) {
        if ($bandwidth <= 0.0) {
            throw new \InvalidArgumentException('MeanShift: bandwidth must be > 0.');
        }
        if (!in_array($kernel, ['flat', 'gaussian'], true)) {
            throw new \InvalidArgumentException(
                "MeanShift: kernel must be 'flat' or 'gaussian'; got '{$kernel}'."
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
     * Assign new samples to the nearest discovered cluster mode.
     *
     * If $cluster_all = false, samples farther than $bandwidth from every
     * mode are labelled −1 (noise).
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples] int cluster labels
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('MeanShift::predict() requires a 2D tensor.');
        }

        [$n, $d]   = $X->shape;
        $nClusters = $this->n_clusters_;
        $bwSq      = $this->bandwidth * $this->bandwidth;

        // Distance from each X[i] to each mode — reuse the BLAS expansion.
        $D      = $this->distanceMatrix($X, $this->cluster_centers_, $n, $nClusters, $d);
        $labels = new Tensor([$n]);

        for ($i = 0; $i < $n; $i++) {
            $minDist = INF;
            $minJ    = -1;
            for ($j = 0; $j < $nClusters; $j++) {
                $dist = (float) $D->buffer[$i * $nClusters + $j];
                if ($dist < $minDist) {
                    $minDist = $dist;
                    $minJ    = $j;
                }
            }

            if (!$this->cluster_all && $minDist > $bwSq) {
                $labels->buffer[$i] = -1.0; // noise
            } else {
                $labels->buffer[$i] = (float) $minJ;
            }
        }
        return $labels;
    }

    // ── Core algorithm ────────────────────────────────────────────────────

    /**
     * Run Mean Shift on X and return per-sample cluster labels.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Ignored (unsupervised).
     * @return Tensor         [n_samples] int cluster labels (−1 = noise if not cluster_all)
     */
    public function fit_predict(Tensor $X, ?Tensor $y = null): Tensor
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'MeanShift: X must be 2-D [n_samples, n_features].'
            );
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;
        $blas                 = BlasEngine::get()->ffi;

        $bwSq       = $this->bandwidth * $this->bandwidth;
        $mergeTol   = $this->cluster_merge_tol ?? ($this->bandwidth * 0.5);
        $mergeTolSq = $mergeTol * $mergeTol;

        // ── Initialise seeds as a copy of X ─────────────────────────────────
        //
        // Seeds[n,d] — each row is a point that will be shifted uphill.
        // Copied in one C call via BLAS scopy.
        $seeds = new Tensor([$n, $d]);
        $blas->cblas_scopy($n * $d, $X->buffer, 1, $seeds->buffer, 1);

        // ── Main Mean Shift loop ─────────────────────────────────────────────
        $iter      = 0;
        $converged = false;

        for ($iter = 1; $iter <= $this->max_iter; $iter++) {
            // ── Step 1: Squared distance matrix D[n,n] — seeds vs X ─────────
            //
            // D[i,j] = ‖seeds[i] − X[j]‖²
            //
            // Using the BLAS expansion with seeds as the "query" matrix and X
            // as the "reference" matrix:
            //   D = snorm ⊗ 1_n + 1_n ⊗ xnorm − 2 · seeds @ X^T
            $snorm = new Tensor([$n]);
            for ($i = 0; $i < $n; $i++) {
                $sPtr              = \FFI::cast('float*', \FFI::addr($seeds->buffer[$i * $d]));
                $snorm->buffer[$i] = $blas->cblas_sdot($d, $sPtr, 1, $sPtr, 1);
            }

            $xnorm = new Tensor([$n]);
            for ($i = 0; $i < $n; $i++) {
                $xPtr              = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
                $xnorm->buffer[$i] = $blas->cblas_sdot($d, $xPtr, 1, $xPtr, 1);
            }

            $D = new Tensor([$n, $n]);
            $blas->cblas_sgemm(101, 111, 112, $n, $n, $d, -2.0,
                $seeds->buffer, $d, $X->buffer, $d, 0.0, $D->buffer, $n);

            $ones_n = Tensor::ones([$n]);
            $blas->cblas_sger(101, $n, $n, 1.0, $snorm->buffer, 1, $ones_n->buffer, 1, $D->buffer, $n);
            $blas->cblas_sger(101, $n, $n, 1.0, $ones_n->buffer, 1, $xnorm->buffer, 1, $D->buffer, $n);

            // ── Step 2: Build weight matrix W[n,n] ───────────────────────────
            //
            // Flat kernel:     W[i,j] = 1        if D[i,j] ≤ bwSq
            // Gaussian kernel: W[i,j] = exp(−D[i,j] / (2·bw²))
            //
            // W is row-normalised in step 3 implicitly via division by counts.
            $W      = new Tensor([$n, $n]);
            $counts = array_fill(0, $n, 0.0);

            if ($this->kernel === 'flat') {
                for ($i = 0; $i < $n; $i++) {
                    for ($j = 0; $j < $n; $j++) {
                        $dist = (float) $D->buffer[$i * $n + $j];
                        if ($dist <= $bwSq) {
                            $W->buffer[$i * $n + $j] = 1.0;
                            $counts[$i]             += 1.0;
                        }
                    }
                }
            } else {
                // Gaussian kernel: W[i,j] = exp(−D[i,j] / (2·bwSq))
                $invTwoBwSq = -1.0 / (2.0 * $bwSq);
                for ($i = 0; $i < $n; $i++) {
                    for ($j = 0; $j < $n; $j++) {
                        $w                       = exp((float) $D->buffer[$i * $n + $j] * $invTwoBwSq);
                        $W->buffer[$i * $n + $j] = (float) $w;
                        $counts[$i]             += $w;
                    }
                }
            }

            unset($D, $snorm, $xnorm, $ones_n);

            // ── Step 3: new_seeds = W @ X / counts ───────────────────────────
            //
            // numerator[n,d] = W[n,n] @ X[n,d]
            //   via sgemm(NoTrans, NoTrans, n, d, n, 1.0, W, n, X, d, 0, num, d)
            //
            // Then divide each row i by counts[i].
            $numerator = new Tensor([$n, $d]);
            $blas->cblas_sgemm(101, 111, 111, $n, $d, $n, 1.0,
                $W->buffer, $n, $X->buffer, $d, 0.0, $numerator->buffer, $d);
            unset($W);

            // Row-normalise and compute max shift for convergence check.
            $maxShift   = 0.0;
            $newSeeds   = new Tensor([$n, $d]);

            for ($i = 0; $i < $n; $i++) {
                $count = max($counts[$i], 1e-12);
                $shift = 0.0;
                for ($l = 0; $l < $d; $l++) {
                    $newVal = (float) $numerator->buffer[$i * $d + $l] / $count;
                    $oldVal = (float) $seeds->buffer[$i * $d + $l];
                    $newSeeds->buffer[$i * $d + $l] = (float) $newVal;
                    $shift += ($newVal - $oldVal) ** 2;
                }
                if ($shift > $maxShift) { $maxShift = $shift; }
            }

            $seeds = $newSeeds;

            // Convergence: max ‖seed_new[i] − seed_old[i]‖ < tol
            if (sqrt($maxShift) < $this->tol) {
                $converged = true;
                break;
            }
        }

        // ── Mode merging: group converged seeds into cluster modes ───────────
        //
        // Two seeds are in the same cluster if ‖s_i − s_j‖ ≤ mergeTol.
        // Greedy algorithm: scan seeds in order; assign to the first existing
        // mode within mergeTolSq, or create a new mode.
        //
        // seedLabel[i] = index of the mode that seed i merged into.
        $modes      = [];   // list of [d] PHP float arrays — mode positions
        $seedLabel  = array_fill(0, $n, -1);

        for ($i = 0; $i < $n; $i++) {
            $si       = [];
            for ($l = 0; $l < $d; $l++) {
                $si[$l] = (float) $seeds->buffer[$i * $d + $l];
            }

            $matched = -1;
            foreach ($modes as $mIdx => $mode) {
                $distSq = 0.0;
                for ($l = 0; $l < $d; $l++) {
                    $diff    = $si[$l] - $mode[$l];
                    $distSq += $diff * $diff;
                }
                if ($distSq <= $mergeTolSq) {
                    $matched = $mIdx;
                    break;
                }
            }

            if ($matched >= 0) {
                $seedLabel[$i] = $matched;
                // Update mode position: running mean of merged seeds.
                // Simple approach: keep the first seed's position as the mode.
                // (Weighted average would require counting — unnecessary for labelling.)
            } else {
                $seedLabel[$i] = count($modes);
                $modes[]       = $si;
            }
        }

        $nClusters = count($modes);

        // ── Store cluster centers Tensor [nClusters, d] ───────────────────────
        $centersTensor = new Tensor([$nClusters, $d]);
        foreach ($modes as $mIdx => $mode) {
            for ($l = 0; $l < $d; $l++) {
                $centersTensor->buffer[$mIdx * $d + $l] = (float) $mode[$l];
            }
        }

        $this->cluster_centers_ = $centersTensor;
        $this->n_clusters_      = $nClusters;
        $this->n_iter_          = $iter;

        // ── Assign original X[i] to the mode that its seed converged to ──────
        //
        // seedLabel[i] is the cluster mode index for seed i (= X[i]).
        // With cluster_all = false, samples farther than bandwidth from their
        // mode are re-labelled −1.
        $labelTensor = new Tensor([$n]);

        for ($i = 0; $i < $n; $i++) {
            if (!$this->cluster_all) {
                // Verify proximity to assigned mode.
                $mIdx   = $seedLabel[$i];
                $distSq = 0.0;
                for ($l = 0; $l < $d; $l++) {
                    $diff    = (float) $seeds->buffer[$i * $d + $l] - $modes[$mIdx][$l];
                    $distSq += $diff * $diff;
                }
                $labelTensor->buffer[$i] = ($distSq <= $bwSq)
                    ? (float) $mIdx
                    : -1.0;
            } else {
                $labelTensor->buffer[$i] = (float) $seedLabel[$i];
            }
        }

        $this->labels_ = $labelTensor;

        return $labelTensor;
    }

    // ── BLAS distance matrix (query [nq,d] vs reference [nr,d]) ──────────

    /**
     * Squared Euclidean distance D[nq, nr] where D[i,j] = ‖Q[i] − R[j]‖².
     *
     * Uses the identity: D = qnorm ⊗ 1_nr + 1_nq ⊗ rnorm − 2 · Q @ R^T
     */
    private function distanceMatrix(Tensor $Q, Tensor $R, int $nq, int $nr, int $d): Tensor
    {
        $blas = BlasEngine::get()->ffi;

        $qnorm = new Tensor([$nq]);
        for ($i = 0; $i < $nq; $i++) {
            $p              = \FFI::cast('float*', \FFI::addr($Q->buffer[$i * $d]));
            $qnorm->buffer[$i] = $blas->cblas_sdot($d, $p, 1, $p, 1);
        }

        $rnorm = new Tensor([$nr]);
        for ($j = 0; $j < $nr; $j++) {
            $p              = \FFI::cast('float*', \FFI::addr($R->buffer[$j * $d]));
            $rnorm->buffer[$j] = $blas->cblas_sdot($d, $p, 1, $p, 1);
        }

        $D = new Tensor([$nq, $nr]);
        $blas->cblas_sgemm(101, 111, 112, $nq, $nr, $d, -2.0,
            $Q->buffer, $d, $R->buffer, $d, 0.0, $D->buffer, $nr);

        $ones_nr = Tensor::ones([$nr]);
        $blas->cblas_sger(101, $nq, $nr, 1.0, $qnorm->buffer, 1, $ones_nr->buffer, 1, $D->buffer, $nr);

        $ones_nq = Tensor::ones([$nq]);
        $blas->cblas_sger(101, $nq, $nr, 1.0, $ones_nq->buffer, 1, $rnorm->buffer, 1, $D->buffer, $nr);

        return $D;
    }

    private function checkFitted(): void
    {
        if (!isset($this->cluster_centers_)) {
            throw new \RuntimeException('MeanShift is not fitted. Call fit() first.');
        }
    }
}
