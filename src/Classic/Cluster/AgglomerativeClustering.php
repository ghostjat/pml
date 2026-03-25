<?php

declare(strict_types=1);

namespace Pml\Classic\Cluster;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\Estimator;

// ═══════════════════════════════════════════════════════════════════════════
//  AgglomerativeClustering — sklearn.cluster.AgglomerativeClustering
//
//  Bottom-up hierarchical clustering with Ward's linkage.
//
//  ── Algorithm ─────────────────────────────────────────────────────────────
//
//  1. Initialise:  N singleton clusters, one per training sample.
//
//  2. Repeat (N − n_clusters) times:
//       a. Find the pair (p, q) of active clusters minimising Ward distance.
//       b. Merge p + q → new cluster r.
//       c. Update all distances from r to remaining clusters using the
//          Lance-Williams recurrence (see below).
//
//  3. Assign output labels 0 … n_clusters−1 to all samples.
//
//  ── Ward's Linkage ─────────────────────────────────────────────────────────
//
//  The Ward merge cost between clusters A and B is the increase in total
//  within-cluster sum-of-squares (SSE) resulting from the merge:
//
//    Δ(A, B)  =  (n_A · n_B) / (n_A + n_B)  ·  ‖centroid_A − centroid_B‖²
//
//  Minimising Δ at each step produces compact, roughly equal-sized clusters.
//
//  ── Lance-Williams Update ──────────────────────────────────────────────────
//
//  After merging clusters p and q into cluster r, the distance from r to any
//  remaining active cluster l is updated without recomputing centroids:
//
//    d(r, l)  =  [(n_l + n_p) · d(p, l)
//                + (n_l + n_q) · d(q, l)
//                −  n_l       · d(p, q)] / (n_l + n_p + n_q)
//
//  This O(n_active) update replaces an O(n_p + n_q) centroid recomputation
//  and avoids storing the full feature matrix beyond fit-time.
//
//  ── Initial Distance Matrix via BLAS ─────────────────────────────────────
//
//  Squared Euclidean ‖x_i − x_j‖² is computed via the identity:
//
//    ‖x_i − x_j‖² = ‖x_i‖² + ‖x_j‖² − 2 x_i · x_j
//
//  Using the same three-step BLAS pattern as DBSCAN:
//
//    1. norms[i] = cblas_sdot(d, X[i,:], 1, X[i,:], 1)      — O(N) sdot calls
//    2. D = −2 X X^T                                          — 1 sgemm call
//    3. D += norms ⊗ ones                                     — 1 sger call
//    4. D += ones ⊗ norms                                     — 1 sger call
//
//  Initial Ward distances: ward(i,j) = 0.5 · D[i,j]
//  (because n_i = n_j = 1 → n_i·n_j/(n_i+n_j) = 0.5).
//
//  The N×N float32 Tensor is materialised and immediately destructed after
//  copying the upper triangle into a flat PHP float[][] working table.
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  Time:   O(N²) BLAS for initial matrix + O(N³) PHP for merge loop
//          (N² merge scans × N active clusters per scan).
//  Memory: O(N²) distance table (decreasing as clusters merge).
//
//  For N ≤ 1 000 this is fast.  N = 5 000 is feasible but slow in PHP.
// ═══════════════════════════════════════════════════════════════════════════

final class AgglomerativeClustering implements Estimator
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Cluster label for each training sample (0 … n_clusters−1).
     * Shape: [n_samples].
     */
    public readonly Tensor $labels_;

    /** Number of clusters found (equals $n_clusters constructor param). */
    public readonly int $n_clusters_;

    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int $n_clusters  Number of clusters to form.
     */
    public function __construct(
        private readonly int $n_clusters = 2,
    ) {
        if ($n_clusters < 1) {
            throw new \InvalidArgumentException(
                'AgglomerativeClustering: n_clusters must be ≥ 1.'
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit the hierarchy and store cluster labels in $this->labels_.
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
     * Cluster $X and return integer cluster labels.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Ignored.
     * @return Tensor         [n_samples] — integer cluster labels 0 … n_clusters−1
     */
    public function fit_predict(Tensor $X, ?Tensor $y = null): Tensor
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'AgglomerativeClustering: X must be 2-D [n_samples, n_features].'
            );
        }

        [$n, $d] = $X->shape;

        if ($this->n_clusters > $n) {
            throw new \InvalidArgumentException(
                "AgglomerativeClustering: n_clusters ({$this->n_clusters}) cannot exceed n_samples ({$n})."
            );
        }

        $this->n_features_in_ = $d;
        $blas                 = BlasEngine::get()->ffi;

        // ── Step 1: Initial squared Euclidean distance matrix via BLAS ────
        //
        // Same three-step expansion as DBSCAN:
        //   D = -2·X·X^T  (sgemm)
        //   D += norms ⊗ ones (sger row norms broadcast)
        //   D += ones ⊗ norms (sger col norms broadcast)
        //
        // Result: D[i,j] = ‖x_i − x_j‖²  (non-negative, up to floating-point error).
        $norms = new Tensor([$n]);
        for ($i = 0; $i < $n; $i++) {
            $rowPtr            = \FFI::cast('float*', \FFI::addr($X->buffer[$i * $d]));
            $norms->buffer[$i] = $blas->cblas_sdot($d, $rowPtr, 1, $rowPtr, 1);
        }

        $D = new Tensor([$n, $n]);
        $blas->cblas_sgemm(101, 111, 112, $n, $n, $d, -2.0, $X->buffer, $d, $X->buffer, $d, 0.0, $D->buffer, $n);

        $ones = Tensor::ones([$n]);
        $blas->cblas_sger(101, $n, $n, 1.0, $norms->buffer, 1, $ones->buffer, 1, $D->buffer, $n);
        $blas->cblas_sger(101, $n, $n, 1.0, $ones->buffer, 1, $norms->buffer, 1, $D->buffer, $n);

        // ── Step 2: Materialise upper-triangle as PHP float[][] ────────────
        //
        // Ward initial distance = 0.5 · ‖x_i − x_j‖²
        // (n_i = n_j = 1  →  n_i·n_j/(n_i+n_j) = 0.5)
        //
        // We only keep [i < j] entries (upper triangle).
        // The Tensor D is freed immediately after this loop.
        $distTable = [];
        for ($i = 0; $i < $n; $i++) {
            for ($j = $i + 1; $j < $n; $j++) {
                $sq = max(0.0, (float) $D->buffer[$i * $n + $j]);
                $distTable[$i][$j] = 0.5 * $sq;
            }
        }

        unset($D, $norms, $ones);  // release N×N Tensor memory immediately

        // ── Step 3: Initialise clustering state ────────────────────────────
        //
        // $active:  hash set of current cluster IDs (initially 0…N-1).
        // $sizes:   int[cluster_id] = number of samples in that cluster.
        // $members: int[][cluster_id] = original sample indices.
        $active  = array_fill_keys(range(0, $n - 1), true);
        $sizes   = array_fill(0, $n, 1);
        $members = [];
        for ($i = 0; $i < $n; $i++) {
            $members[$i] = [$i];
        }
        $nextId = $n;

        // ── Step 4: Greedy Ward merge loop ────────────────────────────────
        //
        // Perform (N − n_clusters) merges.  Each iteration:
        //   a. O(n_active²) scan to find minimum-distance pair (p, q).
        //   b. O(n_active)  Lance-Williams update of distances to new cluster r.
        //   c. O(n_active)  cleanup of old distance entries.
        //
        // Total: O(N³) in the worst case — acceptable for N ≤ 1 000.
        $mergesToDo = $n - $this->n_clusters;

        for ($step = 0; $step < $mergesToDo; $step++) {
            $activeIds = array_keys($active);
            $nActive   = count($activeIds);

            // ── a. Find minimum-distance pair ─────────────────────────────
            $minDist = INF;
            $bestP   = -1;
            $bestQ   = -1;

            for ($ii = 0; $ii < $nActive; $ii++) {
                $p = $activeIds[$ii];
                for ($jj = $ii + 1; $jj < $nActive; $jj++) {
                    $q    = $activeIds[$jj];
                    $lo   = ($p < $q) ? $p : $q;
                    $hi   = ($p < $q) ? $q : $p;
                    $d_pq = $distTable[$lo][$hi] ?? INF;
                    if ($d_pq < $minDist) {
                        $minDist = $d_pq;
                        $bestP   = $p;
                        $bestQ   = $q;
                    }
                }
            }

            $r   = $nextId++;
            $np  = $sizes[$bestP];
            $nq  = $sizes[$bestQ];

            $loP = ($bestP < $bestQ) ? $bestP : $bestQ;
            $hiP = ($bestP < $bestQ) ? $bestQ : $bestP;
            $dPQ = $distTable[$loP][$hiP];

            // ── b. Lance-Williams Ward update for new cluster r ───────────
            //
            // d(r, l) = [(n_l + n_p) d(p,l) + (n_l + n_q) d(q,l) - n_l d(p,q)]
            //            / (n_l + n_p + n_q)
            foreach ($activeIds as $l) {
                if ($l === $bestP || $l === $bestQ) {
                    continue;
                }
                $nl      = $sizes[$l];
                $denom   = $nl + $np + $nq;

                $loPl = ($bestP < $l) ? $bestP : $l;
                $hiPl = ($bestP < $l) ? $l : $bestP;
                $dPL  = $distTable[$loPl][$hiPl] ?? 0.0;

                $loQl = ($bestQ < $l) ? $bestQ : $l;
                $hiQl = ($bestQ < $l) ? $l : $bestQ;
                $dQL  = $distTable[$loQl][$hiQl] ?? 0.0;

                $newDist = (($nl + $np) * $dPL + ($nl + $nq) * $dQL - $nl * $dPQ) / $denom;

                $lo_rl = ($r < $l) ? $r : $l;
                $hi_rl = ($r < $l) ? $l : $r;
                $distTable[$lo_rl][$hi_rl] = max(0.0, $newDist);
            }

            // ── c. Remove stale distance entries ─────────────────────────
            unset($distTable[$loP][$hiP]);

            foreach ($activeIds as $l) {
                if ($l === $bestP || $l === $bestQ) {
                    continue;
                }
                $loPl = ($bestP < $l) ? $bestP : $l;
                $hiPl = ($bestP < $l) ? $l : $bestP;
                unset($distTable[$loPl][$hiPl]);

                $loQl = ($bestQ < $l) ? $bestQ : $l;
                $hiQl = ($bestQ < $l) ? $l : $bestQ;
                unset($distTable[$loQl][$hiQl]);
            }

            // ── d. Update cluster metadata ─────────────────────────────
            $sizes[$r]   = $np + $nq;
            $members[$r] = array_merge($members[$bestP], $members[$bestQ]);

            unset($active[$bestP], $active[$bestQ]);
            $active[$r] = true;
        }

        // ── Step 5: Build output label Tensor ─────────────────────────────
        //
        // Re-label final clusters 0 … n_clusters−1.
        // The ordering is determined by the iteration order of $active (which
        // reflects the merge sequence, consistent with sklearn's behaviour).
        $labelTensor = new Tensor([$n]);
        $clusterIdx  = 0;

        foreach ($active as $clusterId => $_) {
            foreach ($members[$clusterId] as $sampleIdx) {
                $labelTensor->buffer[$sampleIdx] = (float) $clusterIdx;
            }
            $clusterIdx++;
        }

        $this->labels_     = $labelTensor;
        $this->n_clusters_ = $this->n_clusters;

        return $labelTensor;
    }
}
