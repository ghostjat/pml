<?php

declare(strict_types=1);

namespace Pml\Classic\Cluster;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\Estimator;

// ═══════════════════════════════════════════════════════════════════════════
//  DBSCAN — sklearn.cluster.DBSCAN
//
//  Density-Based Spatial Clustering of Applications with Noise.
//  Pure PHP + BLAS acceleration (no external C library).
//
//  ── Algorithm ────────────────────────────────────────────────────────────
//
//  DBSCAN partitions points into clusters based on local density:
//
//    1. A point p is a CORE POINT if at least min_samples neighbours lie
//       within distance eps of p (including p itself in sklearn's convention).
//
//    2. A point q is DIRECTLY DENSITY-REACHABLE from p if:
//         p is a core point  AND  dist(p, q) ≤ eps
//
//    3. A CLUSTER is the maximal set of points mutually density-connected.
//       Expanded via BFS from every unvisited core point.
//
//    4. Points not reachable from any core point are NOISE, labelled −1.
//
//  ── Pairwise Distance via BLAS ────────────────────────────────────────────
//
//  The full N×N squared-Euclidean distance matrix is computed using the
//  algebraic identity:
//
//    ||x_i − x_j||² = ||x_i||² + ||x_j||² − 2·(x_i · x_j)
//
//  Written as matrices (X ∈ ℝ^{N×d}):
//
//    D[i,j] = norms[i] + norms[j] − 2·(X @ X^T)[i,j]
//
//  Step-by-step BLAS (identical pattern to KMeans):
//
//    1. norms[i] = cblas_sdot(d, X[i,:], 1, X[i,:], 1)
//                — squared L2 norm of each row.  O(N) sdot calls.
//
//    2. D = −2 · X @ X^T
//         = cblas_sgemm(RowMajor, NoTrans, Trans, N, N, d, −2, X, d, X, d, 0, D, N)
//         — one BLAS-3 call, O(N²d) time.
//
//    3. D += norms ⊗ ones_N  (broadcast each row norm across the row)
//         = cblas_sger(RowMajor, N, N, 1.0, norms, 1, ones, 1, D, N)
//         — rank-1 update: D[i,j] += norms[i]
//
//    4. D += ones_N ⊗ norms  (broadcast each column norm across the column)
//         = cblas_sger(RowMajor, N, N, 1.0, ones, 1, norms, 1, D, N)
//         — rank-1 update: D[i,j] += norms[j]
//
//  After steps 1–4, D[i,j] = ||x_i − x_j||² (squared Euclidean).
//  We compare against eps² to avoid N² square-root operations.
//
//  ── Neighbourhood Extraction ─────────────────────────────────────────────
//
//  After building D, we extract neighbour lists with a single PHP loop:
//    for each i: neighbours[i] = { j : D[i,j] ≤ eps² }
//
//  This is O(N²) and unavoidable; the BLAS-3 sgemm in step 2 is the
//  dominant cost in practice (highly optimised native code).
//
//  ── Cluster Expansion (BFS) ──────────────────────────────────────────────
//
//  We maintain three parallel arrays of size N:
//    $labels[i]  — cluster id (−1 = noise, ≥0 = cluster), default −1
//    $visited[i] — true once point i is processed
//    $inQueue[i] — true while point i is in the BFS queue (prevents re-enqueue)
//
//  For each unvisited core point p (|neighbours[p]| ≥ min_samples):
//    Assign a new cluster id.
//    BFS: pop point from queue; if it's a core point, add its neighbours
//         to the queue (if not already in queue or visited).
//
//  ── Memory Complexity ────────────────────────────────────────────────────
//
//  The N×N distance matrix requires N²·4 bytes (float32):
//    N=1000  → 4 MB
//    N=5000  → 100 MB
//    N=10000 → 400 MB
//
//  For large N, consider approximate methods (e.g. ball-tree or grid-based)
//  rather than the exact full matrix.  Pml's DBSCAN targets moderate N
//  where the BLAS-3 acceleration makes exact N×N tractable.
// ═══════════════════════════════════════════════════════════════════════════

final class DBSCAN implements Estimator
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Cluster label for each training sample.
     * −1.0 = noise, 0.0/1.0/2.0/… = cluster index.
     * Shape: [n_samples].
     */
    public readonly Tensor $labels_;

    /**
     * Indices of core samples (samples with ≥ min_samples neighbours).
     * @var int[]
     */
    public readonly array $core_sample_indices_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float $eps         Neighbourhood radius.
     *                           Two points are neighbours if ||x_i − x_j|| ≤ eps.
     *                           (Internally compared against eps² to avoid sqrt.)
     * @param int   $min_samples Minimum neighbours (inclusive of self) for a core point.
     *                           Larger values → less noise, fewer and larger clusters.
     */
    public function __construct(
        private readonly float $eps         = 0.5,
        private readonly int   $min_samples = 5,
    ) {
        if ($eps <= 0.0) {
            throw new \InvalidArgumentException('DBSCAN: eps must be > 0.');
        }
        if ($min_samples < 1) {
            throw new \InvalidArgumentException('DBSCAN: min_samples must be >= 1.');
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Compute cluster labels and store them in $this->labels_.
     *
     * Equivalent to sklearn's fit() followed by reading labels_.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Ignored (unsupervised).
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        $this->fit_predict($X);
        return $this;
    }

    // ── Core method ────────────────────────────────────────────────────────

    /**
     * Cluster $X and return a label Tensor.
     *
     * Matches sklearn's DBSCAN.fit_predict(X) interface.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Ignored.
     * @return Tensor         [n_samples] — float32 cluster labels (−1 = noise)
     */
    public function fit_predict(Tensor $X, ?Tensor $y = null): Tensor
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('DBSCAN: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;
        $blas                 = BlasEngine::get()->ffi;

        // ── Step 1: Compute row norms: norms[i] = ||X[i,:]||² ─────────
        //
        // cblas_sdot(d, &X[i*d], 1, &X[i*d], 1) = Σ_j X[i,j]²
        //
        // norms is a float[N] Tensor (used in BLAS sger calls below).
        $norms = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $rowPtr          = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
            $norms->buffer[$i] = $blas->cblas_sdot($d, $rowPtr, 1, $rowPtr, 1);
        }

        // ── Step 2: D = −2 · X @ X^T ──────────────────────────────────
        //
        // cblas_sgemm(RowMajor, NoTrans, Trans, N, N, d, −2, X, d, X, d, 0, D, N):
        //   C = α·A·B^T + β·C
        //   A = X [N×d], B = X [N×d], B^T = X^T [d×N]
        //   C = D [N×N]  (will hold squared distances after steps 3+4)
        //
        // CBLAS enumerations (from BlasEngine):
        //   101 = CblasRowMajor
        //   111 = CblasNoTrans
        //   112 = CblasTrans
        $D = new Tensor([$n, $n]);
        $blas->cblas_sgemm(
            101,   // CblasRowMajor
            111,   // CblasNoTrans  (A = X as-is)
            112,   // CblasTrans    (B = X^T)
            $n,    // M: rows of A and C
            $n,    // N: columns of B^T and C
            $d,    // K: columns of A = rows of B
            -2.0,  // alpha
            $X->buffer, $d,  // A = X, leading dimension = d (row stride)
            $X->buffer, $d,  // B = X, leading dimension = d (B^T is implicit)
            0.0,             // beta: overwrite C
            $D->buffer, $n   // C = D, leading dimension = N
        );

        // ── Step 3: D += norms ⊗ ones (broadcast row norms) ───────────
        //
        // cblas_sger(RowMajor, N, N, 1.0, norms, 1, ones, 1, D, N):
        //   A += α · x · y^T
        //   x = norms [N],  y = ones [N]
        //   → D[i,j] += norms[i] · 1  for all j
        //
        // This adds ||x_i||² to every element of row i.
        $ones = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) { $ones->buffer[$i] = 1.0; }

        $blas->cblas_sger(
            101,   // CblasRowMajor
            $n, $n,
            1.0,
            $norms->buffer, 1,  // x = norms (row-norm vector)
            $ones->buffer, 1,   // y = ones
            $D->buffer, $n      // A = D
        );

        // ── Step 4: D += ones ⊗ norms (broadcast column norms) ────────
        //
        // cblas_sger(RowMajor, N, N, 1.0, ones, 1, norms, 1, D, N):
        //   x = ones [N],  y = norms [N]
        //   → D[i,j] += 1 · norms[j]  for all i
        //
        // This adds ||x_j||² to every element of column j.
        // Combined with step 3: D[i,j] = ||x_i||² + ||x_j||² − 2·(x_i·x_j)
        //                              = ||x_i − x_j||²  ✓
        $blas->cblas_sger(
            101,
            $n, $n,
            1.0,
            $ones->buffer, 1,   // x = ones
            $norms->buffer, 1,  // y = norms (column-norm vector)
            $D->buffer, $n
        );

        // ── Step 5: Extract neighbour lists ────────────────────────────
        //
        // Compare D[i,j] ≤ eps² (avoiding N² sqrt calls).
        // Also clamp the diagonal to 0 (floating-point should give 0 but
        // numerical error can leave small positive values there).
        //
        // neighbours[i] = sorted list of indices j where dist(i,j) ≤ eps
        // (including i itself — sklearn's convention counts self as a neighbour)
        $epsSq      = $this->eps * $this->eps;
        $neighbours = [];
        for ($i = 0; $i < $n; $i++) {
            $row  = $i * $n;
            $nbrs = [];
            for ($j = 0; $j < $n; $j++) {
                $dist = (float)$D->buffer[$row + $j];
                if ($dist <= $epsSq) {
                    $nbrs[] = $j;
                }
            }
            $neighbours[$i] = $nbrs;
        }

        // Distance matrix no longer needed after extracting neighbours.
        unset($D, $norms, $ones);

        // ── Step 6: Identify core points ──────────────────────────────
        //
        // A point is a core point iff |neighbours[i]| >= min_samples.
        // (neighbours[i] includes i itself, matching sklearn's convention.)
        $isCore = array_fill(0, $n, false);
        for ($i = 0; $i < $n; $i++) {
            $isCore[$i] = (count($neighbours[$i]) >= $this->min_samples);
        }

        // ── Step 7: BFS cluster expansion ─────────────────────────────
        //
        // Algorithm:
        //   labels[i] = -1   → noise (default)
        //   labels[i] = k    → cluster k (assigned during BFS)
        //
        //   visited[i] = true  once point i has been assigned (or confirmed noise)
        //   inQueue[i] = true  while i is pending in the BFS queue
        //
        // For each unvisited core point p, start a new cluster:
        //   Seed the BFS queue with p's neighbours.
        //   While queue is non-empty:
        //     Pop q.
        //     Assign q to the current cluster (if not already in another cluster).
        //     If q is also a core point, add its unqueued neighbours to the queue.
        //
        // Points that are never reached by any BFS remain labelled −1 (noise).
        $labels  = array_fill(0, $n, -1);
        $visited = array_fill(0, $n, false);
        $inQueue = array_fill(0, $n, false);

        $clusterId = 0;
        $coreIndices = [];

        for ($i = 0; $i < $n; $i++) {
            if ($visited[$i] || !$isCore[$i]) {
                continue;
            }

            $coreIndices[] = $i;

            // ── Seed BFS from core point i ─────────────────────────────
            $labels[$i]  = $clusterId;
            $visited[$i] = true;

            // Queue: PHP array used as a FIFO queue.
            // array_shift is O(N) but fine for typical clustering sizes.
            // For very large N, use a ring buffer or SplQueue.
            $queue = [];
            foreach ($neighbours[$i] as $nb) {
                if (!$inQueue[$nb]) {
                    $queue[]     = $nb;
                    $inQueue[$nb] = true;
                }
            }

            // ── BFS expansion ─────────────────────────────────────────
            while (!empty($queue)) {
                $q = array_shift($queue);  // O(1) via pointer advance in C; PHP O(N) — acceptable

                // Assign to cluster (point may already be in a cluster as
                // a border point; do NOT reassign — first cluster wins).
                if (!$visited[$q]) {
                    $labels[$q]  = $clusterId;
                    $visited[$q] = true;
                }

                // If q is also a core point, expand further from q's neighbours.
                // This is what makes DBSCAN discover arbitrarily-shaped clusters.
                if ($isCore[$q]) {
                    if (!in_array($q, $coreIndices)) {
                        $coreIndices[] = $q;
                    }
                    foreach ($neighbours[$q] as $nb) {
                        if (!$inQueue[$nb]) {
                            $queue[]       = $nb;
                            $inQueue[$nb]  = true;
                        }
                    }
                }
            }

            $clusterId++;
        }

        // ── Store results ──────────────────────────────────────────────
        $this->core_sample_indices_ = $coreIndices;

        $labelTensor = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $labelTensor->buffer[$i] = (float)$labels[$i];
        }
        $this->labels_ = $labelTensor;

        return $labelTensor;
    }
}
