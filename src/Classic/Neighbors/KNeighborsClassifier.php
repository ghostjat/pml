<?php

declare(strict_types=1);

namespace Pml\Classic\Neighbors;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  KNeighborsClassifier — sklearn.neighbors.KNeighborsClassifier
//
//  Lazy (instance-based) classifier: fit() just stores training data.
//  predict() computes exact Euclidean distances to all training samples
//  using the BLAS distance expansion trick, then takes a majority vote
//  over the k nearest neighbours.
//
//  ── BLAS Distance Expansion ──────────────────────────────────────────────
//
//  Exact same method used in KMeans::distanceMatrix():
//
//    ||x_test_i − x_train_j||² = ||x_test_i||² + ||x_train_j||² − 2 x_test_i · x_train_j
//
//  Written as matrices (X_test [n_t, d], X_train [n_r, d]):
//
//    D [n_t, n_r] = tnorm [n_t, 1]  +  rnorm [1, n_r]  −  2 · (X_test @ X_train^T)
//
//  BLAS steps:
//    1. tnorm[i] = sdot(d, X_test[i,:],  1, X_test[i,:],  1)  ← n_t sdot calls
//    2. rnorm[j] = sdot(d, X_train[j,:], 1, X_train[j,:], 1)  ← n_r sdot calls
//    3. D        = −2 · sgemm(X_test, X_train^T)               ← single BLAS-3 call
//    4. D       += tnorm ⊗ ones_nr  (sger outer-product broadcast)
//    5. D       += ones_nt ⊗ rnorm  (sger outer-product broadcast)
//
//  ── k-NN Selection ───────────────────────────────────────────────────────
//
//  After computing D [n_test, n_train], for each test sample i we need the
//  indices of the k smallest distances in row D[i, :].
//
//  Strategy: PHP SplPriorityQueue (max-heap) of capacity k.
//    - Insert each (index, distance) with priority = distance.
//    - After every insert, if count > k, extract() removes the current
//      maximum (farthest neighbour), maintaining exactly k candidates.
//    - Complexity: O(n_train · log k) per test sample vs O(n_train · log n_train)
//      for a full sort — meaningful speedup when k << n_train.
//
//  ── Majority Vote ────────────────────────────────────────────────────────
//
//  After finding the k nearest train indices, we tally their labels.
//  Ties are broken by the smallest class label (consistent with sklearn's
//  stable sort behaviour for the 'uniform' weights mode).
// ═══════════════════════════════════════════════════════════════════════════

final class KNeighborsClassifier implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** Training feature matrix — stored as-is (lazy / instance-based). */
    public readonly Tensor $_fit_X;

    /** Training label vector [n_samples_fit] — float32 encoding of int labels. */
    public readonly Tensor $_fit_y;

    /** Unique class labels discovered in fit(), sorted ascending. */
    public readonly array $classes_;

    public readonly int $n_features_in_;
    public readonly int $n_samples_fit_;

    /**
     * @param int    $n_neighbors  Number of nearest neighbours k.
     * @param string $weights      'uniform' (equal votes) — 'distance' weighting
     *                             is not yet implemented.
     * @param string $metric       Distance metric.  Only 'euclidean' is supported.
     */
    public function __construct(
        private readonly int    $n_neighbors = 5,
        private readonly string $weights     = 'uniform',
        private readonly string $metric      = 'euclidean',
    ) {
        if ($n_neighbors < 1) {
            throw new \InvalidArgumentException("KNeighborsClassifier: n_neighbors must be ≥ 1.");
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Store training data.  KNN is a lazy learner — no model is built here.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Class labels   [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('KNeighborsClassifier::fit() requires labels $y.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('KNeighborsClassifier::fit() requires a 2-D X.');
        }

        [$n, $d] = $X->shape;

        if ($n < $this->n_neighbors) {
            throw new \InvalidArgumentException(
                "KNeighborsClassifier: n_samples={$n} < n_neighbors={$this->n_neighbors}."
            );
        }

        // Discover unique class labels (sorted)
        $seen = [];
        for ($i = 0; $i < $n; $i++) {
            $seen[(int) round((float) $y->buffer[$i])] = true;
        }
        ksort($seen);

        $this->_fit_X         = $X;
        $this->_fit_y         = $y;
        $this->classes_       = array_keys($seen);
        $this->n_features_in_ = $d;
        $this->n_samples_fit_ = $n;

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Classify test samples by majority vote of their k nearest training neighbours.
     *
     * @param Tensor $X  Test feature matrix [n_test, n_features]
     * @return Tensor    Predicted labels     [n_test]
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "KNeighborsClassifier::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        [$n_t, $d] = $X->shape;
        $n_r       = $this->n_samples_fit_;
        $k         = $this->n_neighbors;

        // ── Compute full D [n_t, n_r] via BLAS expansion ──────────────────
        $D = $this->distanceMatrix($X, $this->_fit_X, $n_t, $n_r, $d);

        // ── Per test sample: k-NN selection + majority vote ────────────────
        $out = new Tensor([$n_t]);

        for ($i = 0; $i < $n_t; $i++) {
            $rowOffset = $i * $n_r;

            // ── SplPriorityQueue max-heap of capacity k ────────────────────
            //
            // We use priority = distance so the queue's natural max-extraction
            // removes the FARTHEST candidate on every overflow, leaving the k
            // NEAREST in the heap.  This is O(n_r · log k) per test sample.
            $heap = new \SplPriorityQueue();

            for ($j = 0; $j < $n_r; $j++) {
                // max(0, …) guards against tiny negative values from floating
                // point cancellation in the distance expansion.
                $dist = max(0.0, (float) $D->buffer[$rowOffset + $j]);

                // Insert training index $j with priority = $dist.
                // SplPriorityQueue is a max-heap → largest priority extracted first.
                $heap->insert($j, $dist);

                // Overflow: drop the current farthest candidate
                if ($heap->count() > $k) {
                    $heap->extract();
                }
            }

            // ── Tally votes from the k remaining (nearest) candidates ──────
            $votes = [];
            while (!$heap->isEmpty()) {
                $idx   = $heap->extract(); // training sample index
                $label = (int) round((float) $this->_fit_y->buffer[$idx]);
                $votes[$label] = ($votes[$label] ?? 0) + 1;
            }

            // Sort by vote count descending; ties broken by label (ksort order
            // preserved because arsort is stable in PHP 8).
            // Smallest-label-wins on tie: sort ascending by label first, then
            // descending by count using arsort (stable in PHP 8 ≥ 8.0).
            ksort($votes);
            arsort($votes);

            $out->buffer[$i] = (float) array_key_first($votes);
        }

        return $out;
    }

    /**
     * Return accuracy on test data: fraction of correct predictions.
     * Mirrors sklearn's ClassifierMixin.score().
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

    // ── BLAS distance matrix ──────────────────────────────────────────────

    /**
     * Compute D [n_t, n_r] where D[i,j] = ||X_test[i] − X_train[j]||²
     *
     * Identical algorithm to KMeans::distanceMatrix() — see that class for
     * the full mathematical derivation.
     *
     *   D = tnorm ⊗ ones_nr  +  ones_nt ⊗ rnorm  −  2 · (X_test @ X_train^T)
     */
    private function distanceMatrix(
        Tensor $Xt, Tensor $Xr,
        int    $n_t, int $n_r, int $d
    ): Tensor {
        $blas = BlasEngine::get()->ffi;

        // ── Row norms of test set ||x_test_i||² ───────────────────────────
        $tnorm = new Tensor([$n_t]);
        for ($i = 0; $i < $n_t; $i++) {
            $ptr           = \FFI::cast('float*', \FFI::addr($Xt->buffer[$i * $d]));
            $tnorm->buffer[$i] = (float) $blas->cblas_sdot($d, $ptr, 1, $ptr, 1);
        }

        // ── Row norms of train set ||x_train_j||² ─────────────────────────
        $rnorm = new Tensor([$n_r]);
        for ($j = 0; $j < $n_r; $j++) {
            $ptr           = \FFI::cast('float*', \FFI::addr($Xr->buffer[$j * $d]));
            $rnorm->buffer[$j] = (float) $blas->cblas_sdot($d, $ptr, 1, $ptr, 1);
        }

        // ── D = −2 · X_test @ X_train^T  (one BLAS-3 call) ───────────────
        //
        //  sgemm(RowMajor, NoTrans, Trans, n_t, n_r, d,
        //        −2.0, X_test[n_t,d], d, X_train[n_r,d], d, 0.0, D[n_t,n_r], n_r)
        //
        //  D[i,j] = −2 · Σ_l X_test[i,l] · X_train[j,l]
        $D = new Tensor([$n_t, $n_r]);
        $blas->cblas_sgemm(
            101,            // CblasRowMajor
            111,            // CblasNoTrans — X_test is [n_t, d]
            112,            // CblasTrans   — X_train is [n_r, d], transposed to [d, n_r]
            $n_t, $n_r, $d,
            -2.0,
            $Xt->buffer, $d,
            $Xr->buffer, $d,
            0.0,
            $D->buffer, $n_r
        );

        // ── D += tnorm ⊗ ones_nr  (broadcast test norms across columns) ───
        //
        //  sger(RowMajor, n_t, n_r, 1.0, tnorm[n_t], 1, ones[n_r], 1, D, n_r)
        //  D[i,j] += tnorm[i]  for all j.
        $ones_nr = Tensor::ones([$n_r]);
        $blas->cblas_sger(
            101, $n_t, $n_r, 1.0,
            $tnorm->buffer, 1,
            $ones_nr->buffer, 1,
            $D->buffer, $n_r
        );

        // ── D += ones_nt ⊗ rnorm  (broadcast train norms across rows) ─────
        //
        //  sger(RowMajor, n_t, n_r, 1.0, ones[n_t], 1, rnorm[n_r], 1, D, n_r)
        //  D[i,j] += rnorm[j]  for all i.
        $ones_nt = Tensor::ones([$n_t]);
        $blas->cblas_sger(
            101, $n_t, $n_r, 1.0,
            $ones_nt->buffer, 1,
            $rnorm->buffer, 1,
            $D->buffer, $n_r
        );

        return $D;
    }

    private function checkFitted(): void
    {
        if (!isset($this->_fit_X)) {
            throw new \RuntimeException('KNeighborsClassifier is not fitted. Call fit() first.');
        }
    }
}
