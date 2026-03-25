<?php

declare(strict_types=1);

namespace Pml\Classic\Cluster;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\Estimator;

// ═══════════════════════════════════════════════════════════════════════════
//  DBSCAN — sklearn.cluster.DBSCAN
//
//  Density-Based Spatial Clustering of Applications with Noise.
//  Pure PHP + BLAS acceleration — no external spatial index required.
//
//  ── Algorithm ────────────────────────────────────────────────────────────
//
//  DBSCAN partitions points into clusters based on local density:
//
//    1. A point p is a CORE POINT if at least min_samples neighbours lie
//       within distance eps of p (sklearn convention: p counts itself).
//
//    2. A point q is DIRECTLY DENSITY-REACHABLE from p if p is a core
//       point and dist(p, q) ≤ eps.
//
//    3. A CLUSTER is the maximal set of mutually density-connected points,
//       expanded via BFS from every unvisited core point.
//
//    4. Points unreachable from any core point are NOISE, labelled −1.
//
//  ── Pairwise Distance via BLAS ────────────────────────────────────────────
//
//  Full N×N squared-Euclidean distance matrix via the expansion identity:
//
//    ||x_i − x_j||² = ||x_i||² + ||x_j||² − 2·(x_i · x_j)
//
//  Step-by-step BLAS (identical pattern to KMeans):
//
//    1. norms[i] = cblas_sdot(d, X[i,:], 1, X[i,:], 1)   — O(N) sdot calls
//    2. D = −2 · X @ X^T      — single sgemm(RowMajor, NoTrans, Trans, N, N, d, −2, …)
//    3. D += norms ⊗ ones_N   — sger rank-1: D[i,j] += norms[i]
//    4. D += ones_N ⊗ norms   — sger rank-1: D[i,j] += norms[j]
//
//  Neighbours are extracted with a single PHP loop: dist(i,j) ≤ eps²
//  (avoiding N² sqrt calls).
//
//  ── BFS Queue ────────────────────────────────────────────────────────────
//
//  Uses PHP's SplQueue (doubly-linked list, O(1) enqueue + dequeue) instead
//  of array_shift (O(N) per dequeue on plain arrays).  Core-point membership
//  is tracked via a hash set ($coreSet) for O(1) lookup.
//
//  ── Memory ───────────────────────────────────────────────────────────────
//
//  N×N float32 matrix:  N=1000 → 4 MB  ·  N=5000 → 100 MB  ·  N=10000 → 400 MB
//  The distance matrix is freed immediately after extracting neighbour lists.
//
//  ── KDTree Note ──────────────────────────────────────────────────────────
//
//  When Pml\Classic\Spatial\KDTree becomes available, the $O(N^2)$ neighbour
//  extraction can be replaced with KDTree::query_radius() to achieve
//  $O(N \log N)$ average complexity.  The BFS and clustering logic below
//  remain unchanged — only the neighbours[] array construction needs updating.
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
     * Indices of core samples (those with ≥ min_samples neighbours).
     * @var int[]
     */
    public readonly array $core_sample_indices_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param float $eps         Neighbourhood radius.
     *                           Two points are neighbours when ||x_i − x_j|| ≤ eps.
     *                           Internally compared against eps² to avoid sqrt.
     * @param int   $min_samples Minimum neighbours (inclusive of self) for a core point.
     *                           Larger values → fewer, larger clusters and more noise.
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

    // ── Estimator ─────────────────────────────────────────────────────────

    /**
     * Compute cluster labels and store them in $this->labels_.
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
     * Cluster $X and return a Tensor of integer cluster labels.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Ignored.
     * @return Tensor         [n_samples] float32 — cluster labels (−1 = noise)
     */
    public function fit_predict(Tensor $X, ?Tensor $y = null): Tensor
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('DBSCAN: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;
        $blas                 = BlasEngine::get()->ffi;

        // ── Step 1: Row norms:  norms[i] = ||X[i,:]||² ─────────────────────
        $norms = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $rowPtr            = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
            $norms->buffer[$i] = $blas->cblas_sdot($d, $rowPtr, 1, $rowPtr, 1);
        }

        // ── Step 2: D = −2 · X @ X^T  (one BLAS-3 sgemm call) ─────────────
        $D = new Tensor([$n, $n]);
        $blas->cblas_sgemm(
            101,  // CblasRowMajor
            111,  // CblasNoTrans (A = X as-is)
            112,  // CblasTrans   (B = X^T)
            $n, $n, $d,
            -2.0,
            $X->buffer, $d,
            $X->buffer, $d,
            0.0,
            $D->buffer, $n
        );

        // ── Step 3: D += norms ⊗ ones — broadcast row norms ─────────────────
        //   sger: D[i,j] += norms[i] · 1  for all j
        $ones = Tensor::ones([$n]);
        $blas->cblas_sger(101, $n, $n, 1.0, $norms->buffer, 1, $ones->buffer, 1, $D->buffer, $n);

        // ── Step 4: D += ones ⊗ norms — broadcast column norms ──────────────
        //   sger: D[i,j] += 1 · norms[j]  for all i
        //   Combined with step 3: D[i,j] = ||x_i − x_j||²  ✓
        $blas->cblas_sger(101, $n, $n, 1.0, $ones->buffer, 1, $norms->buffer, 1, $D->buffer, $n);

        // ── Step 5: Extract neighbour lists from D ───────────────────────────
        //
        // Compare D[i,j] ≤ eps² — avoids N² sqrt operations.
        // Self-distance D[i,i] may have small floating-point error; the
        // eps > 0 check ensures i is always its own neighbour (sklearn convention).
        $epsSq      = $this->eps * $this->eps;
        $neighbours = [];
        for ($i = 0; $i < $n; $i++) {
            $row  = $i * $n;
            $nbrs = [];
            for ($j = 0; $j < $n; $j++) {
                if ((float) $D->buffer[$row + $j] <= $epsSq) {
                    $nbrs[] = $j;
                }
            }
            $neighbours[$i] = $nbrs;
        }

        // Free the N×N matrix immediately — no longer needed.
        unset($D, $norms, $ones);

        // ── Step 6: Identify core points ─────────────────────────────────────
        //
        // A point is a core point iff |neighbours[i]| >= min_samples.
        // Using a hash set ($coreSet) for O(1) lookup in BFS.
        $isCore  = array_fill(0, $n, false);
        $coreSet = [];   // hash set of core-point indices: $coreSet[$i] = true

        for ($i = 0; $i < $n; $i++) {
            if (count($neighbours[$i]) >= $this->min_samples) {
                $isCore[$i]  = true;
                $coreSet[$i] = true;
            }
        }

        // ── Step 7: BFS cluster expansion ────────────────────────────────────
        //
        // State arrays:
        //   $labels[$i]   — cluster id (−1 = noise), default −1
        //   $visited[$i]  — true once point i is fully processed
        //   $inQueue[$i]  — true while i is pending in the BFS queue
        //
        // SplQueue provides O(1) enqueue (push) and dequeue (shift).
        // The $inQueue flag prevents redundant enqueuing.
        //
        // For each unvisited core point:
        //   Assign a new cluster id.
        //   Seed BFS with its neighbours.
        //   Expand: if a dequeued point is also a core point, add its
        //           unqueued neighbours to the queue.
        $labels      = array_fill(0, $n, -1);
        $visited     = array_fill(0, $n, false);
        $inQueue     = array_fill(0, $n, false);
        $coreIndices = [];
        $clusterId   = 0;

        for ($i = 0; $i < $n; $i++) {
            if ($visited[$i] || !$isCore[$i]) {
                continue;
            }

            $coreIndices[] = $i;
            $labels[$i]    = $clusterId;
            $visited[$i]   = true;

            // ── Seed BFS queue with all neighbours of core point i ───────────
            $queue = new \SplQueue();

            foreach ($neighbours[$i] as $nb) {
                if (!$inQueue[$nb]) {
                    $queue->enqueue($nb);
                    $inQueue[$nb] = true;
                }
            }

            // ── BFS expansion ────────────────────────────────────────────────
            while (!$queue->isEmpty()) {
                $q = $queue->dequeue();   // O(1) — SplQueue (doubly-linked list)

                // Assign q to the current cluster if not yet visited.
                // Border points may already belong to another cluster (first-wins).
                if (!$visited[$q]) {
                    $labels[$q]  = $clusterId;
                    $visited[$q] = true;
                }

                // If q is a core point, expand the frontier through its neighbours.
                if (isset($coreSet[$q])) {
                    if (!in_array($q, $coreIndices, true)) {
                        $coreIndices[] = $q;
                    }
                    foreach ($neighbours[$q] as $nb) {
                        if (!$inQueue[$nb]) {
                            $queue->enqueue($nb);
                            $inQueue[$nb] = true;
                        }
                    }
                }
            }

            $clusterId++;
        }

        // ── Store fitted state ────────────────────────────────────────────────
        $this->core_sample_indices_ = $coreIndices;

        $labelTensor = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $labelTensor->buffer[$i] = (float) $labels[$i];
        }
        $this->labels_ = $labelTensor;

        return $labelTensor;
    }
}
