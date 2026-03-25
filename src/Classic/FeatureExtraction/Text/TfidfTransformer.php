<?php

declare(strict_types=1);

namespace Pml\Classic\FeatureExtraction\Text;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Transformer};
use Pml\Classic\Preprocess\Normalizer;

// ═══════════════════════════════════════════════════════════════════════════
//  TfidfTransformer — sklearn.feature_extraction.text.TfidfTransformer
//
//  Transforms a raw term-frequency (count) matrix into a TF-IDF matrix:
//
//    tfidf[i, j]  =  tf[i, j]  ×  idf[j]
//
//  then optionally applies row-wise L1 or L2 normalisation.
//
//  ── IDF formula (sklearn smoothed, default) ────────────────────────────────
//
//    idf[j] = log( (1 + N) / (1 + df[j]) ) + 1
//
//  where N = number of training documents and df[j] = number of documents
//  in which term j appears at least once.  Adding 1 to numerator and
//  denominator (smooth_idf=true) prevents IDF from being 0 for very common
//  terms and avoids division-by-zero for unseen terms at inference time.
//
//  With smooth_idf=false the raw formula is used:
//
//    idf[j] = log( N / df[j] ) + 1
//
//  ── BLAS strategy ─────────────────────────────────────────────────────────
//
//  IDF scaling: for each column j, the scalar idf[j] multiplies every
//  element in column j.  In a row-major matrix, column j lives at offsets
//  j, j+n_features, j+2·n_features, … so cblas_sscal with stride=n_features
//  scales the entire column in one C call — the standard BLAS trick for
//  column-wise operations on row-major storage (same as StandardScaler).
//
//    for j in 0…n_features-1:
//        sscal(n_docs, idf[j], &out[0,j], stride=n_features)
//
//  Row-normalisation is delegated to Normalizer (L1 / L2 / max).
//
//  ── DF computation ────────────────────────────────────────────────────────
//
//  During fit(), we count the number of non-zero entries per column.
//  A PHP loop over the flat buffer is unavoidable here (we need a threshold
//  comparison "> 0"), but it runs once and is O(N × V) in PHP integer
//  arithmetic — typically fast enough for vocabulary sizes up to ~100k.
// ═══════════════════════════════════════════════════════════════════════════

final class TfidfTransformer implements Estimator, Transformer
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Inverse document frequency vector.
     * Shape: [n_features].  idf_[j] = log((1+N)/(1+df[j])) + 1 (smoothed).
     */
    public readonly Tensor $idf_;

    public readonly int $n_features_in_;

    /** Number of documents seen during fit(). */
    public readonly int $n_samples_fit_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param bool   $use_idf    Weight TF counts by IDF (default true).
     * @param string $norm       Row-wise normalisation: 'l1', 'l2', or 'none'.
     * @param bool   $smooth_idf Add 1 to DF numerator/denominator (default true).
     */
    public function __construct(
        private readonly bool   $use_idf    = true,
        private readonly string $norm       = 'l2',
        private readonly bool   $smooth_idf = true,
    ) {
        if (!in_array($norm, ['l1', 'l2', 'none'], true)) {
            throw new \InvalidArgumentException(
                "TfidfTransformer: norm must be 'l1', 'l2', or 'none'; got '{$norm}'."
            );
        }
    }

    // ── Estimator ─────────────────────────────────────────────────────────

    /**
     * Learn IDF weights from the training count matrix.
     *
     * @param Tensor      $X  Count matrix [n_docs, n_features]
     * @param Tensor|null $y  Ignored.
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'TfidfTransformer::fit() requires a 2-D Tensor [n_docs, n_features].'
            );
        }

        [$n, $nf] = $X->shape;

        // ── Document frequency: df[j] = #{docs with tf[i,j] > 0} ─────────
        //
        // Iterate the flat buffer once.  Pure PHP integer arithmetic;
        // no FFI overhead per element since we only read via cast to float.
        $df = array_fill(0, $nf, 0);
        for ($i = 0; $i < $n; $i++) {
            $base = $i * $nf;
            for ($j = 0; $j < $nf; $j++) {
                if ((float) $X->buffer[$base + $j] > 0.0) {
                    $df[$j]++;
                }
            }
        }

        // ── IDF vector ─────────────────────────────────────────────────────
        $idf = new Tensor([$nf]);

        if ($this->use_idf) {
            if ($this->smooth_idf) {
                // log( (1+N) / (1+df[j]) ) + 1  — sklearn default
                for ($j = 0; $j < $nf; $j++) {
                    $idf->buffer[$j] = log((1.0 + $n) / (1.0 + $df[$j])) + 1.0;
                }
            } else {
                // log( N / df[j] ) + 1  — unsmoothed; guard df=0 with max()
                for ($j = 0; $j < $nf; $j++) {
                    $idf->buffer[$j] = log($n / max(1, $df[$j])) + 1.0;
                }
            }
        } else {
            // use_idf=false: IDF is all-ones (TF only)
            $bytes = pack('f*', ...array_fill(0, $nf, 1.0));
            \FFI::memcpy($idf->buffer, $bytes, $nf * 4);
        }

        $this->idf_           = $idf;
        $this->n_features_in_ = $nf;
        $this->n_samples_fit_ = $n;

        return $this;
    }

    // ── Transformer ───────────────────────────────────────────────────────

    /**
     * Apply TF-IDF weighting and row normalisation to a count matrix.
     *
     * @param Tensor $X  Count matrix [n_docs, n_features]
     * @return Tensor    TF-IDF matrix [n_docs, n_features]
     */
    public function transform(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "TfidfTransformer::transform() expected [*, {$this->n_features_in_}], "
                . "got [" . implode(', ', $X->shape) . "]."
            );
        }

        [$n, $nf] = $X->shape;
        $blas     = BlasEngine::get()->ffi;
        $out      = $X->clone();

        // ── IDF column scaling ─────────────────────────────────────────────
        //
        // sscal with stride $nf: starting at out[0,j], steps over n_features
        // elements at a time, landing on out[1,j], out[2,j], … — the entire
        // column j.  One C call per feature.
        if ($this->use_idf) {
            for ($j = 0; $j < $nf; $j++) {
                $colPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$j]));
                $blas->cblas_sscal($n, (float) $this->idf_->buffer[$j], $colPtr, $nf);
            }
        }

        // ── Row normalisation ──────────────────────────────────────────────
        if ($this->norm !== 'none') {
            $normalizer = new Normalizer($this->norm);
            $out        = $normalizer->transform($out);
        }

        return $out;
    }

    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->fit($X, $y)->transform($X);
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private function checkFitted(): void
    {
        if (!isset($this->idf_)) {
            throw new \RuntimeException(
                'TfidfTransformer is not fitted yet. Call fit() before transform().'
            );
        }
    }
}
