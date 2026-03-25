<?php

declare(strict_types=1);

namespace Pml\Classic\Preprocess;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Transformer};

// ═══════════════════════════════════════════════════════════════════════════
//  Normalizer — sklearn.preprocessing.Normalizer
//
//  Scales each SAMPLE (row) independently to have unit norm.
//  Stateless: fit() is a no-op; transform() requires no prior fitting.
//
//  ── Norms supported ───────────────────────────────────────────────────────
//
//    'l1'  : x̂_i = x_i / Σ|x_i|          (taxicab / Manhattan norm)
//    'l2'  : x̂_i = x_i / √(Σ x_i²)       (Euclidean norm)  [default]
//    'max' : x̂_i = x_i / max(|x_i|)       (Chebyshev norm)
//
//  All-zero rows are left unchanged (norm = 0 guard).
//
//  ── BLAS strategy ─────────────────────────────────────────────────────────
//
//    L1  : cblas_sasum(n, row, 1)      → sum of |x_i| in one C call
//    L2  : cblas_sdot(n, row, 1, …)   → Σ x_i² in one C call; then sqrt()
//    max : cblas_isamax(n, row, 1)     → index of max |x_i|; read value
//
//    Scaling: cblas_sscal(n, 1/norm, row, 1) — one C call per row.
//
//  O(m) FFI calls, each O(n) in C — no heavy float math in PHP.
// ═══════════════════════════════════════════════════════════════════════════

final class Normalizer implements Estimator, Transformer
{
    /**
     * @param string $norm  'l1' | 'l2' | 'max'
     */
    public function __construct(
        private readonly string $norm = 'l2',
    ) {
        if (!in_array($norm, ['l1', 'l2', 'max'], true)) {
            throw new \InvalidArgumentException(
                "Normalizer: norm must be 'l1', 'l2', or 'max'; got '{$norm}'."
            );
        }
    }

    // ── Estimator ─────────────────────────────────────────────────────────

    /**
     * Stateless: no parameters to fit.  Always returns $this unchanged.
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        return $this;
    }

    // ── Transformer ───────────────────────────────────────────────────────

    /**
     * Normalise each row of $X to unit norm.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples, n_features] — normalised copy
     */
    public function transform(Tensor $X): Tensor
    {
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException(
                'Normalizer::transform() requires a 2-D tensor [n_samples, n_features].'
            );
        }

        [$m, $n] = $X->shape;
        $blas    = BlasEngine::get()->ffi;
        $out     = $X->clone();

        for ($i = 0; $i < $m; $i++) {
            $rowPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $n]));

            $norm = match ($this->norm) {
                // ── L1: sum of absolute values ─────────────────────────────
                //   cblas_sasum returns Σ|x_k| — one C call for the whole row
                'l1'  => (float) $blas->cblas_sasum($n, $rowPtr, 1),

                // ── L2: Euclidean norm ─────────────────────────────────────
                //   sdot(row, row) = Σ x_k² ; then sqrt() in PHP (one scalar op)
                'l2'  => sqrt(max(0.0, (float) $blas->cblas_sdot($n, $rowPtr, 1, $rowPtr, 1))),

                // ── Max: Chebyshev norm ────────────────────────────────────
                //   isamax → index of |x_k|_max; read the absolute value
                'max' => abs((float) $out->buffer[$i * $n + (int) $blas->cblas_isamax($n, $rowPtr, 1)]),
            };

            if ($norm < 1e-14) {
                continue;   // all-zero row: leave unchanged (sklearn behaviour)
            }

            // scale row in-place: row *= 1/norm
            $blas->cblas_sscal($n, 1.0 / $norm, $rowPtr, 1);
        }

        return $out;
    }

    public function fit_transform(Tensor $X, ?Tensor $y = null): Tensor
    {
        return $this->transform($X);
    }
}
