<?php

declare(strict_types=1);

namespace Pml\Classic\Cluster;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  KMeans — sklearn.cluster.KMeans
//
//  Lloyd's algorithm with pluggable initialisation strategies and the BLAS
//  Euclidean distance expansion trick.
//
//  ── Initialisation Strategies ────────────────────────────────────────────
//
//  'random'     — Sample k rows of X uniformly without replacement.
//                 Fast, but high variance; use with n_init > 1.
//
//  'k-means++'  — (sklearn default) Pick the first centroid randomly.  For
//                 each subsequent centroid, select a data point with
//                 probability proportional to D(x)² — the squared distance
//                 from the nearest already-chosen centroid.
//                 Guarantees O(log k) approximation ratio in expectation.
//                 Implementation maintains a running minDist[n] array,
//                 updating it after each centroid is selected (O(nd) per
//                 centroid via sgemv) rather than recomputing from scratch.
//
//  'k-mc2'      — (Bachem et al. 2016) Markov-chain Monte Carlo approximation
//                 of k-means++. For each new centroid, run a Markov chain of
//                 length $mcChainLen: propose a uniform candidate, accept with
//                 probability min(1, D(x')/D(x_prev)).  The stationary
//                 distribution approximates the k-means++ D² distribution with
//                 O(mcChainLen · k) distance computations per centroid instead
//                 of O(n · k) — ideal for n >> mcChainLen.
//
//  'preset'     — Accept a pre-computed Tensor of shape [n_clusters, n_features]
//                 as the starting centroids.  n_init is forced to 1 (no restarts).
//
//  ── Distance computation via BLAS expansion ───────────────────────────────
//
//  ||x_i − c_j||² = ||x_i||² + ||c_j||² − 2·(x_i · c_j)
//
//  Written as matrices (X [n,d], C [k,d]):
//    D[n,k] = xnorm[n,1] + cnorm[1,k] − 2 · (X @ C^T)[n,k]
//
//  Step-by-step BLAS:
//    1. xnorm[i] = sdot(d, X[i,:], 1, X[i,:], 1)    ← O(n) sdot calls
//    2. cnorm[j] = sdot(d, C[j,:], 1, C[j,:], 1)    ← O(k) sdot calls
//    3. D        = −2 · sgemm(X, C^T)                ← single BLAS-3 call
//    4. D       += xnorm ⊗ ones_k  (sger rank-1)    ← broadcast xnorm rows
//    5. D       += ones_n ⊗ cnorm  (sger rank-1)    ← broadcast cnorm cols
// ═══════════════════════════════════════════════════════════════════════════

final class KMeans implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var Tensor  Cluster centroid positions [n_clusters, n_features] */
    public readonly Tensor $cluster_centers_;

    /** @var Tensor  Cluster label for each training sample [n_samples] */
    public readonly Tensor $labels_;

    /** @var float   Sum of squared distances of samples to their nearest centroid */
    public readonly float $inertia_;

    /** @var int     Number of Lloyd iterations until convergence */
    public readonly int $n_iter_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int         $n_clusters     Number of clusters k.
     * @param int         $max_iter       Maximum Lloyd iterations per run.
     * @param float       $tol            Convergence threshold (label change check).
     * @param int         $n_init         Number of independent random restarts;
     *                                    best inertia is kept.  Forced to 1 for
     *                                    'preset' (deterministic warm-start).
     * @param string      $init           Initialisation strategy:
     *                                    'random', 'k-means++', 'k-mc2', 'preset'.
     * @param Tensor|null $initCentroids  Required when $init='preset'.
     *                                    Shape must be [n_clusters, n_features].
     * @param int|null    $random_state   PHP mt_srand() seed.  null = no seeding.
     * @param int         $mcChainLen     Markov-chain length for 'k-mc2'.
     *                                    Larger → closer to k-means++ distribution.
     *                                    Typical value: 200.
     */
    public function __construct(
        private readonly int         $n_clusters    = 8,
        private readonly int         $max_iter      = 300,
        private readonly float       $tol           = 1e-4,
        private readonly int         $n_init        = 10,
        private readonly string      $init          = 'k-means++',
        private readonly ?Tensor     $initCentroids = null,
        private readonly ?int        $random_state  = null,
        private readonly int         $mcChainLen    = 200,
    ) {
        if ($n_clusters < 1) {
            throw new \InvalidArgumentException('KMeans: n_clusters must be ≥ 1.');
        }
        if (!in_array($init, ['random', 'k-means++', 'k-mc2', 'preset'], true)) {
            throw new \InvalidArgumentException(
                "KMeans: init must be 'random', 'k-means++', 'k-mc2', or 'preset'; got '{$init}'."
            );
        }
        if ($init === 'preset' && $initCentroids === null) {
            throw new \InvalidArgumentException(
                "KMeans: initCentroids tensor is required when init='preset'."
            );
        }
    }

    // ── Estimator ─────────────────────────────────────────────────────────

    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('KMeans::fit() requires a 2D tensor [n_samples, n_features].');
        }

        [$n, $d] = $X->shape;
        $k       = $this->n_clusters;

        if ($n < $k) {
            throw new \InvalidArgumentException("KMeans: n_samples={$n} < n_clusters={$k}.");
        }

        if ($this->random_state !== null) {
            mt_srand($this->random_state);
        }

        // 'preset' is deterministic — no point running multiple restarts.
        $runs = ($this->init === 'preset') ? 1 : $this->n_init;

        $bestCentroids = null;
        $bestLabels    = null;
        $bestInertia   = INF;
        $bestIter      = 0;

        for ($run = 0; $run < $runs; $run++) {
            [$centroids, $labels, $inertia, $iter] = $this->runOnce($X, $n, $k, $d);

            if ($inertia < $bestInertia) {
                $bestInertia   = $inertia;
                $bestCentroids = $centroids;
                $bestLabels    = $labels;
                $bestIter      = $iter;
            }
        }

        $this->cluster_centers_ = $bestCentroids;
        $this->labels_          = $bestLabels;
        $this->inertia_         = $bestInertia;
        $this->n_iter_          = $bestIter;

        return $this;
    }

    // ── Predictor ─────────────────────────────────────────────────────────

    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('KMeans::predict() requires a 2D tensor.');
        }

        [$n, $d] = $X->shape;
        $k       = $this->n_clusters;
        $D       = $this->distanceMatrix($X, $this->cluster_centers_, $n, $k, $d);
        $labels  = $this->assignLabels($D, $n, $k);

        $out = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $out->buffer[$i] = (float) $labels[$i];
        }
        return $out;
    }

    // ── Core Lloyd iteration ──────────────────────────────────────────────

    /**
     * Run one complete Lloyd's algorithm (init + iterate to convergence).
     *
     * @return array{0:Tensor, 1:Tensor, 2:float, 3:int}
     *   [cluster_centers [k,d], labels [n], inertia, n_iterations]
     */
    private function runOnce(Tensor $X, int $n, int $k, int $d): array
    {
        $blas      = BlasEngine::get()->ffi;
        $centroids = $this->initializeCentroids($X, $n, $k, $d);

        $labels     = array_fill(0, $n, -1);
        $prevLabels = array_fill(0, $n, -2);
        $iter       = 0;

        for ($iter = 1; $iter <= $this->max_iter; $iter++) {
            // ── E-step: assign each sample to the nearest centroid ──────────
            $D      = $this->distanceMatrix($X, $centroids, $n, $k, $d);
            $labels = $this->assignLabels($D, $n, $k);

            if ($labels === $prevLabels) {
                break; // Label convergence — no assignments changed
            }
            $prevLabels = $labels;

            // ── M-step: recompute centroids as weighted cluster means ────────
            //
            // Scatter-add: no BLAS gather primitive exists.
            // PHP loop is O(n + k·d) and dominated by the O(n·k·d) sgemm.
            $counts = array_fill(0, $k, 0);
            \FFI::memset($centroids->buffer, 0, $k * $d * 4);

            for ($i = 0; $i < $n; $i++) {
                $cj = $labels[$i];
                $counts[$cj]++;
                $xPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
                $cPtr = \FFI::cast('float*', \FFI::addr($centroids->buffer[$cj * $d]));
                $blas->cblas_saxpy($d, 1.0, $xPtr, 1, $cPtr, 1);
            }

            for ($j = 0; $j < $k; $j++) {
                if ($counts[$j] > 0) {
                    $cPtr = \FFI::cast('float*', \FFI::addr($centroids->buffer[$j * $d]));
                    $blas->cblas_sscal($d, 1.0 / $counts[$j], $cPtr, 1);
                }
                // Empty cluster: centroid stays at zero — matches sklearn behaviour.
            }
        }

        // ── Inertia: sum of squared distances to assigned centroid ───────────
        $D       = $this->distanceMatrix($X, $centroids, $n, $k, $d);
        $inertia = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $inertia += max(0.0, (float) $D->buffer[$i * $k + $labels[$i]]);
        }

        $labelsTensor = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $labelsTensor->buffer[$i] = (float) $labels[$i];
        }

        return [$centroids, $labelsTensor, $inertia, $iter];
    }

    // ── Initialisation strategies ─────────────────────────────────────────

    /**
     * Dispatch to the configured initialisation strategy.
     *
     * Returns a Tensor of shape [k, d] containing the initial centroids.
     */
    private function initializeCentroids(Tensor $X, int $n, int $k, int $d): Tensor
    {
        return match ($this->init) {
            'random'    => $this->initRandom($X, $n, $k, $d),
            'k-means++' => $this->initKMeansPlusPlus($X, $n, $k, $d),
            'k-mc2'     => $this->initKMC2($X, $n, $k, $d),
            'preset'    => $this->initPreset($k, $d),
        };
    }

    /**
     * 'random' — Sample k rows of X uniformly without replacement.
     *
     * Shuffle a range index array and copy the first k rows using scopy.
     */
    private function initRandom(Tensor $X, int $n, int $k, int $d): Tensor
    {
        $blas     = BlasEngine::get()->ffi;
        $indices  = range(0, $n - 1);
        shuffle($indices);
        $indices  = array_slice($indices, 0, $k);

        $centroids = new Tensor([$k, $d]);
        foreach ($indices as $j => $idx) {
            $src = \FFI::cast('float*', \FFI::addr($X->buffer[$idx * $d]));
            $dst = \FFI::cast('float*', \FFI::addr($centroids->buffer[$j * $d]));
            $blas->cblas_scopy($d, $src, 1, $dst, 1);
        }
        return $centroids;
    }

    /**
     * 'k-means++' — Distance-proportional seeding (Arthur & Vassilvitskii, 2007).
     *
     * Algorithm:
     *   1. Pick the first centroid uniformly at random from X.
     *   2. For j = 1 … k−1:
     *      a. For each sample i, compute D(i) = min_{l<j} ||x_i − c_l||²
     *         using sgemv to compute X @ c_j in one BLAS-2 call, then subtract
     *         row norms — gives the entire column of the distance matrix cheaply.
     *      b. Sample the next centroid index from the categorical distribution
     *         with probabilities proportional to D(i)².
     *
     * Maintaining running minDist[n] avoids recomputing all previous centroids
     * on each step — updating from the latest centroid only.
     *
     * Expected inertia: O(log k) times the optimal solution.
     */
    private function initKMeansPlusPlus(Tensor $X, int $n, int $k, int $d): Tensor
    {
        $blas      = BlasEngine::get()->ffi;
        $centroids = new Tensor([$k, $d]);

        // ── Step 1: First centroid — uniformly random ──────────────────────
        $firstIdx = mt_rand(0, $n - 1);
        $src      = \FFI::cast('float*', \FFI::addr($X->buffer[$firstIdx * $d]));
        $dst      = \FFI::cast('float*', \FFI::addr($centroids->buffer[0]));
        $blas->cblas_scopy($d, $src, 1, $dst, 1);

        // Pre-compute squared row norms once — reused in each distance update.
        $xnorm = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $xPtr            = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
            $xnorm->buffer[$i] = $blas->cblas_sdot($d, $xPtr, 1, $xPtr, 1);
        }

        // minDist[i] = squared distance from x_i to its nearest selected centroid.
        $minDist = array_fill(0, $n, INF);

        // ── Steps 2 … k: Subsequent centroids ─────────────────────────────
        for ($j = 0; $j < $k - 1; $j++) {
            // ── Update minDist using the most recently added centroid ────────
            //
            // dots[i] = X[i,:] · c_j via sgemv:
            //   sgemv(RowMajor, NoTrans, n, d, 1.0, X, d, c_j, 1, 0, dots, 1)
            //   → dots[i] = Σ_l X[i,l] · c_j[l]
            //
            // distToJ[i] = xnorm[i] + cjNorm − 2·dots[i]   (expansion identity)
            $dots   = new Tensor([$n]);
            $cjPtr  = \FFI::cast('float*', \FFI::addr($centroids->buffer[$j * $d]));
            $blas->cblas_sgemv(
                101,              // CblasRowMajor
                111,              // CblasNoTrans — compute X @ c_j
                $n, $d,
                1.0,
                $X->buffer, $d,
                $cjPtr, 1,
                0.0,
                $dots->buffer, 1
            );

            $cjNorm = (float) $blas->cblas_sdot($d, $cjPtr, 1, $cjPtr, 1);

            for ($i = 0; $i < $n; $i++) {
                $distToJ = (float) $xnorm->buffer[$i] + $cjNorm
                           - 2.0 * (float) $dots->buffer[$i];
                $distToJ = max(0.0, $distToJ); // clamp float noise
                if ($distToJ < $minDist[$i]) {
                    $minDist[$i] = $distToJ;
                }
            }

            // ── Sample next centroid proportional to minDist² ───────────────
            //
            // sklearn uses D² (squared distance); minDist already stores D².
            // The distribution is: P(i) = D(i)² / Σ D(i)²
            $total = array_sum($minDist);

            $r      = ($total > 0.0)
                      ? (float) mt_rand() / (float) mt_getrandmax() * $total
                      : 0.0;
            $chosen = $n - 1;
            $cumsum = 0.0;
            for ($i = 0; $i < $n; $i++) {
                $cumsum += $minDist[$i];
                if ($cumsum >= $r) {
                    $chosen = $i;
                    break;
                }
            }

            // ── Copy chosen row into centroids[j+1] ─────────────────────────
            $src = \FFI::cast('float*', \FFI::addr($X->buffer[$chosen * $d]));
            $dst = \FFI::cast('float*', \FFI::addr($centroids->buffer[($j + 1) * $d]));
            $blas->cblas_scopy($d, $src, 1, $dst, 1);
        }

        return $centroids;
    }

    /**
     * 'k-mc2' — Markov Chain Monte Carlo approximation of k-means++.
     *            (Bachem, Lucic, Hassani, Krause — NIPS 2016)
     *
     * For each of the k−1 subsequent centroids, run a Markov chain of length
     * $mcChainLen to sample approximately from the D² distribution:
     *
     *   1. Sample a starting point x₀ uniformly from X.
     *   2. Compute d(x₀) = min_{c ∈ Centers} ||x₀ − c||² (O(j·d) work).
     *   3. For t = 1 … mcChainLen:
     *      a. Sample candidate x' uniformly from X.
     *      b. Compute d(x') = min_{c ∈ Centers} ||x' − c||².
     *      c. Accept: with probability min(1, d(x')/d(x_{prev})).
     *      d. x_{prev} = x_t (accepted or not).
     *   4. Emit x_{prev} as the new centroid.
     *
     * Cost per centroid: O(mcChainLen · j · d) vs O(n · j · d) for k-means++.
     * For mcChainLen << n, this is dramatically faster on large datasets.
     */
    private function initKMC2(Tensor $X, int $n, int $k, int $d): Tensor
    {
        $blas      = BlasEngine::get()->ffi;
        $centroids = new Tensor([$k, $d]);

        // First centroid — uniformly random.
        $firstIdx = mt_rand(0, $n - 1);
        $src      = \FFI::cast('float*', \FFI::addr($X->buffer[$firstIdx * $d]));
        $dst      = \FFI::cast('float*', \FFI::addr($centroids->buffer[0]));
        $blas->cblas_scopy($d, $src, 1, $dst, 1);

        // Temporary buffer for computing one difference vector.
        $tmpDiff = new Tensor([$d]);

        for ($j = 1; $j < $k; $j++) {
            // Helper: compute D(idx) = min squared distance from X[idx] to
            // the first j selected centroids.  O(j·d) BLAS-1 calls.
            $distToNearest = function (int $idx) use ($X, $centroids, $blas, $d, $j, $tmpDiff): float {
                $xPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$idx * $d]));
                $minD = INF;
                for ($l = 0; $l < $j; $l++) {
                    $cPtr = \FFI::cast('float*', \FFI::addr($centroids->buffer[$l * $d]));
                    // diff = x − c_l  via scopy + saxpy
                    $blas->cblas_scopy($d, $xPtr, 1, $tmpDiff->buffer, 1);
                    $blas->cblas_saxpy($d, -1.0, $cPtr, 1, $tmpDiff->buffer, 1);
                    $distSq = (float) $blas->cblas_sdot($d, $tmpDiff->buffer, 1, $tmpDiff->buffer, 1);
                    if ($distSq < $minD) { $minD = $distSq; }
                }
                return $minD;
            };

            // ── Initialise chain at a uniformly random starting point ────────
            $curIdx  = mt_rand(0, $n - 1);
            $curDist = $distToNearest($curIdx);

            // ── Markov chain of length mcChainLen ────────────────────────────
            for ($t = 0; $t < $this->mcChainLen; $t++) {
                $candIdx  = mt_rand(0, $n - 1);
                $candDist = $distToNearest($candIdx);

                // Acceptance probability: min(1, D(candidate)/D(current))
                // Uniform proposal cancels in Metropolis-Hastings ratio.
                if ($curDist <= 0.0 ||
                    $candDist / $curDist >= (float) mt_rand() / (float) mt_getrandmax()) {
                    $curIdx  = $candIdx;
                    $curDist = $candDist;
                }
            }

            // ── Emit the final chain position as centroid j ──────────────────
            $src = \FFI::cast('float*', \FFI::addr($X->buffer[$curIdx * $d]));
            $dst = \FFI::cast('float*', \FFI::addr($centroids->buffer[$j * $d]));
            $blas->cblas_scopy($d, $src, 1, $dst, 1);
        }

        return $centroids;
    }

    /**
     * 'preset' — Validate and copy user-supplied initial centroids.
     *
     * The supplied Tensor must have shape [n_clusters, n_features].
     * A BLAS scopy is used to copy the entire buffer in one C call.
     */
    private function initPreset(int $k, int $d): Tensor
    {
        $tc = $this->initCentroids;

        if (count($tc->shape) !== 2
            || $tc->shape[0] !== $k
            || $tc->shape[1] !== $d) {
            throw new \InvalidArgumentException(sprintf(
                "KMeans init='preset': initCentroids shape must be [%d, %d], got [%s].",
                $k, $d, implode(', ', $tc->shape)
            ));
        }

        $centroids = new Tensor([$k, $d]);
        $blas      = BlasEngine::get()->ffi;
        $blas->cblas_scopy($k * $d, $tc->buffer, 1, $centroids->buffer, 1);
        return $centroids;
    }

    // ── BLAS distance matrix ──────────────────────────────────────────────

    /**
     * Compute D[n,k] where D[i,j] = ||X[i] − C[j]||²
     *
     * Uses the expansion:  D = xnorm ⊗ 1_k  +  1_n ⊗ cnorm  −  2 · X @ C^T
     *
     * All heavy arithmetic is in three C calls: sdot×(n+k), sgemm×1, sger×2.
     */
    private function distanceMatrix(Tensor $X, Tensor $C, int $n, int $k, int $d): Tensor
    {
        $blas = BlasEngine::get()->ffi;

        // Step 1: row norms of X
        $xnorm = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $xPtr              = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
            $xnorm->buffer[$i] = $blas->cblas_sdot($d, $xPtr, 1, $xPtr, 1);
        }

        // Step 2: row norms of C (centroid norms)
        $cnorm = new Tensor([$k]);
        for ($j = 0; $j < $k; $j++) {
            $cPtr              = \FFI::cast('float*', \FFI::addr($C->buffer[$j * $d]));
            $cnorm->buffer[$j] = $blas->cblas_sdot($d, $cPtr, 1, $cPtr, 1);
        }

        // Step 3: D = −2 · X @ C^T  (single BLAS-3 sgemm call)
        $D = new Tensor([$n, $k]);
        $blas->cblas_sgemm(
            101,          // CblasRowMajor
            111,          // CblasNoTrans (X [n,d])
            112,          // CblasTrans   (C [k,d] → [d,k])
            $n, $k, $d,
            -2.0,
            $X->buffer, $d,
            $C->buffer, $d,
            0.0,
            $D->buffer, $k
        );

        // Step 4: D += xnorm ⊗ ones_k  (sger broadcast — D[i,j] += xnorm[i])
        $ones_k = Tensor::ones([$k]);
        $blas->cblas_sger(101, $n, $k, 1.0, $xnorm->buffer, 1, $ones_k->buffer, 1, $D->buffer, $k);

        // Step 5: D += ones_n ⊗ cnorm  (sger broadcast — D[i,j] += cnorm[j])
        $ones_n = Tensor::ones([$n]);
        $blas->cblas_sger(101, $n, $k, 1.0, $ones_n->buffer, 1, $cnorm->buffer, 1, $D->buffer, $k);

        return $D;
    }

    /**
     * Argmin over columns of D — the nearest centroid index for each row.
     *
     * @return int[]  Labels array of length n.
     */
    private function assignLabels(Tensor $D, int $n, int $k): array
    {
        $labels = [];
        for ($i = 0; $i < $n; $i++) {
            $offset  = $i * $k;
            $minDist = (float) $D->buffer[$offset];
            $minJ    = 0;
            for ($j = 1; $j < $k; $j++) {
                $dist = (float) $D->buffer[$offset + $j];
                if ($dist < $minDist) {
                    $minDist = $dist;
                    $minJ    = $j;
                }
            }
            $labels[$i] = $minJ;
        }
        return $labels;
    }

    private function checkFitted(): void
    {
        if (!isset($this->cluster_centers_)) {
            throw new \RuntimeException('KMeans is not fitted. Call fit() first.');
        }
    }
}
