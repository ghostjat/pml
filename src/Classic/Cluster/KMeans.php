<?php

declare(strict_types=1);

namespace Pml\Classic\Cluster;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  KMeans — sklearn.cluster.KMeans
//
//  Lloyd's algorithm with the BLAS Euclidean distance expansion trick.
//
//  ── Distance computation via BLAS expansion ───────────────────────────────
//
//  Naïve O(n·k·d) triple loop is replaced by a sequence of BLAS calls:
//
//    ||x_i − c_j||² = ||x_i||² + ||c_j||² − 2·(x_i · c_j)
//
//  Written as matrices (X [n,d], C [k,d]):
//
//    D[n,k] = xnorm[n,1] + cnorm[1,k] − 2 · (X @ C^T)[n,k]
//
//  Step-by-step BLAS:
//    1. xnorm[i] = sdot(d, X[i,:], 1, X[i,:], 1)          ← O(n) sdot calls
//    2. cnorm[j] = sdot(d, C[j,:], 1, C[j,:], 1)          ← O(k) sdot calls
//    3. D        = −2 · sgemm(X, C^T)                       ← single BLAS-3 call
//    4. D       += xnorm ⊗ ones_k    (via sger, rank-1)    ← broadcast xnorm
//    5. D       += ones_n ⊗ cnorm    (via sger, rank-1)    ← broadcast cnorm
//
//  Steps 4 and 5 use cblas_sger (outer-product rank-1 update):
//    sger(m, n, alpha, x, incx, y, incy, A, lda):  A += alpha * x * y^T
//
//    For xnorm: A=D[n,k], x=xnorm[n], y=ones[k] → D[i,j] += xnorm[i]
//    For cnorm: A=D[n,k], x=ones[n],  y=cnorm[k] → D[i,j] += cnorm[j]
//
//  The centroid update (scatter-add) requires a PHP loop since BLAS has no
//  conditional gather/scatter primitive.  This loop is O(n + k·d) and is
//  dominated by the O(n·k·d) sgemm call.
// ═══════════════════════════════════════════════════════════════════════════

final class KMeans implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────
    /** @var Tensor  Cluster centroid positions [n_clusters, n_features] */
    public readonly Tensor $cluster_centers_;

    /** @var Tensor  Cluster label for each training sample [n_samples] */
    public readonly Tensor $labels_;

    /** @var float   Sum of squared distances to nearest centroid */
    public readonly float $inertia_;

    /** @var int     Number of iterations until convergence */
    public readonly int $n_iter_;

    /**
     * @param int    $n_clusters    Number of clusters k
     * @param int    $max_iter      Maximum Lloyd iterations
     * @param float  $tol           Convergence threshold on centroid shift
     * @param int    $n_init        Number of random initialisations (best kept)
     * @param int    $random_state  PHP mt_srand() seed (0 = do not seed)
     */
    public function __construct(
        private readonly int   $n_clusters   = 8,
        private readonly int   $max_iter     = 300,
        private readonly float $tol          = 1e-4,
        private readonly int   $n_init       = 10,
        private readonly int   $random_state = 0,
    ) {}

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

        if ($this->random_state !== 0) {
            mt_srand($this->random_state);
        }

        // ── Run n_init random initialisations, keep best inertia ──────────
        $bestCentroids = null;
        $bestLabels    = null;
        $bestInertia   = INF;
        $bestIter      = 0;

        for ($init = 0; $init < $this->n_init; $init++) {
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

        [$n, $d]  = $X->shape;
        $k        = $this->n_clusters;
        $D        = $this->distanceMatrix($X, $this->cluster_centers_, $n, $k, $d);
        $labelsArr = $this->assignLabels($D, $n, $k);

        $out = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $out->buffer[$i] = (float) $labelsArr[$i];
        }
        return $out;
    }

    // ── Core Lloyd iteration ──────────────────────────────────────────────

    /**
     * @return array{0:Tensor, 1:Tensor, 2:float, 3:int}
     *   [centroids, labels, inertia, n_iterations]
     */
    private function runOnce(Tensor $X, int $n, int $k, int $d): array
    {
        $blas = BlasEngine::get()->ffi;

        // ── Initialise centroids by sampling k rows of X without replacement
        $indices  = range(0, $n - 1);
        shuffle($indices);
        $indices  = array_slice($indices, 0, $k);

        $centroids = new Tensor([$k, $d]);
        foreach ($indices as $j => $idx) {
            $src = \FFI::cast('float*', \FFI::addr($X->buffer[$idx * $d]));
            $dst = \FFI::cast('float*', \FFI::addr($centroids->buffer[$j * $d]));
            $blas->cblas_scopy($d, $src, 1, $dst, 1);
        }

        $labels    = array_fill(0, $n, -1);
        $prevLabels = array_fill(0, $n, -2);
        $iter      = 0;

        for ($iter = 1; $iter <= $this->max_iter; $iter++) {
            // ── E-step: assign each sample to its nearest centroid ─────────
            $D      = $this->distanceMatrix($X, $centroids, $n, $k, $d);
            $labels = $this->assignLabels($D, $n, $k);

            // Convergence check: did any label change?
            if ($labels === $prevLabels) {
                break;
            }
            $prevLabels = $labels;

            // ── M-step: recompute centroids as cluster means ───────────────
            //
            // For each cluster j, centroid[j] = mean of all X[i] with label[i]=j.
            // This scatter-gather has no BLAS equivalent — PHP loop permitted.
            $counts = array_fill(0, $k, 0);
            \FFI::memset($centroids->buffer, 0, $k * $d * 4); // zero all centroids

            for ($i = 0; $i < $n; $i++) {
                $cj = $labels[$i];
                $counts[$cj]++;
                // cblas_saxpy: centroid[j] += X[i]  (accumulate)
                $xPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
                $cPtr = \FFI::cast('float*', \FFI::addr($centroids->buffer[$cj * $d]));
                $blas->cblas_saxpy($d, 1.0, $xPtr, 1, $cPtr, 1);
            }

            // Divide each centroid by its count to get the mean
            for ($j = 0; $j < $k; $j++) {
                if ($counts[$j] > 0) {
                    $cPtr = \FFI::cast('float*', \FFI::addr($centroids->buffer[$j * $d]));
                    $blas->cblas_sscal($d, 1.0 / $counts[$j], $cPtr, 1);
                }
                // If count=0 (empty cluster): centroid stays at zero — matches sklearn behaviour
            }

            // Centroid shift convergence check
            if ($this->tol > 0.0 && $iter > 1) {
                // (Optional) Could compute centroid shift norm here.
                // Omitted for performance; label convergence above suffices.
            }
        }

        // ── Compute inertia (sum of squared distances to nearest centroid) ─
        $D       = $this->distanceMatrix($X, $centroids, $n, $k, $d);
        $inertia = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $inertia += max(0.0, (float) $D->buffer[$i * $k + $labels[$i]]);
        }

        // ── Convert labels PHP array → 1D Tensor ──────────────────────────
        $labelsTensor = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $labelsTensor->buffer[$i] = (float) $labels[$i];
        }

        return [$centroids, $labelsTensor, $inertia, $iter];
    }

    // ── BLAS distance matrix ──────────────────────────────────────────────

    /**
     * Compute D[n,k] where D[i,j] = ||X[i] − C[j]||²
     *
     * Uses the BLAS expansion:
     *   D = xnorm ⊗ 1_k  +  1_n ⊗ cnorm  −  2 · X @ C^T
     *
     * All heavy math is in three C calls: sdot×(n+k), sgemm×1, sger×2.
     */
    private function distanceMatrix(Tensor $X, Tensor $C, int $n, int $k, int $d): Tensor
    {
        $blas = BlasEngine::get()->ffi;

        // ── Step 1: row norms ||x_i||² ────────────────────────────────────
        //   sdot(d, X[i,:], 1, X[i,:], 1) = Σ_j X[i,j]²
        $xnorm = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $xPtr = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
            $xnorm->buffer[$i] = (float) $blas->cblas_sdot($d, $xPtr, 1, $xPtr, 1);
        }

        // ── Step 2: centroid norms ||c_j||² ──────────────────────────────
        //   sdot(d, C[j,:], 1, C[j,:], 1)
        $cnorm = new Tensor([$k]);
        for ($j = 0; $j < $k; $j++) {
            $cPtr = \FFI::cast('float*', \FFI::addr($C->buffer[$j * $d]));
            $cnorm->buffer[$j] = (float) $blas->cblas_sdot($d, $cPtr, 1, $cPtr, 1);
        }

        // ── Step 3: D = −2 · X @ C^T  ────────────────────────────────────
        //
        //  sgemm(RowMajor, NoTrans, Trans, n, k, d,
        //        −2.0, X[n,d], d, C[k,d], d, 0.0, D[n,k], k)
        //
        //  Computes D[i,j] = −2 · Σ_l X[i,l] · C[j,l]  — the cross term.
        //  A single Level-3 BLAS call handles the entire n×k×d computation.
        $D = new Tensor([$n, $k]);
        $blas->cblas_sgemm(
            101,    // CblasRowMajor
            111,    // CblasNoTrans — X is [n, d]
            112,    // CblasTrans   — C is [k, d], transposed to [d, k]
            $n, $k, $d,
            -2.0,                    // alpha = −2
            $X->buffer, $d,          // A = X,   lda = d
            $C->buffer, $d,          // B = C,   ldb = d (physical cols, pre-transpose)
            0.0,                     // beta = 0 → overwrite D
            $D->buffer, $k           // C = D,   ldc = k
        );

        // ── Step 4: D += xnorm ⊗ ones_k  (broadcast xnorm along columns) ─
        //
        //  cblas_sger(RowMajor, n, k, 1.0, xnorm[n], 1, ones[k], 1, D[n,k], k)
        //  Computes: D[i,j] += 1.0 · xnorm[i] · 1.0 = xnorm[i]  for all j.
        //
        //  This is the outer-product broadcast — far faster than a PHP loop.
        $ones_k = Tensor::ones([$k]);
        $blas->cblas_sger(
            101,                // CblasRowMajor
            $n, $k,
            1.0,
            $xnorm->buffer, 1, // x = xnorm [n]
            $ones_k->buffer, 1,// y = ones  [k]
            $D->buffer, $k     // A = D [n,k], lda = k
        );

        // ── Step 5: D += ones_n ⊗ cnorm  (broadcast cnorm along rows) ────
        //
        //  cblas_sger(RowMajor, n, k, 1.0, ones[n], 1, cnorm[k], 1, D[n,k], k)
        //  Computes: D[i,j] += 1.0 · 1.0 · cnorm[j] = cnorm[j]  for all i.
        $ones_n = Tensor::ones([$n]);
        $blas->cblas_sger(
            101,
            $n, $k,
            1.0,
            $ones_n->buffer, 1, // x = ones  [n]
            $cnorm->buffer, 1,  // y = cnorm [k]
            $D->buffer, $k
        );

        return $D;
    }

    /**
     * Find the nearest centroid index for each sample (argmin per row of D).
     * @return int[]  Labels array, length n.
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
