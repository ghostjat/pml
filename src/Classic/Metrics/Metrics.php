<?php

declare(strict_types=1);

namespace Pml\Classic\Metrics;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  Metrics — sklearn.metrics (accuracy_score, mean_squared_error, r2_score)
//
//  All three metrics use BLAS Level-1 primitives where possible, routing
//  the bulk of arithmetic to C:
//
//    accuracy_score:
//      Pure PHP comparison loop — no BLAS equivalent for element-wise
//      equality.  O(n) with no allocation beyond the loop counter.
//
//    mean_squared_error:
//      diff  = y_true − y_pred     (cblas_scopy + cblas_saxpy)
//      MSE   = sdot(diff, diff)/n  (single cblas_sdot call)
//
//    r2_score:
//      SS_res = sdot(y_true − y_pred, …)  (same as MSE, no division)
//      SS_tot = sdot(y_true − ȳ,    …)
//      ȳ is computed with a PHP loop (BLAS has no mean primitive);
//      the centred vector is produced with cblas_saxpy on a ones buffer.
//      R² = 1 − SS_res / SS_tot
// ═══════════════════════════════════════════════════════════════════════════

final class Metrics
{
    // ── Classification ─────────────────────────────────────────────────────

    /**
     * Fraction of correctly classified samples.
     *
     * Both tensors are treated as integer labels (rounded, then compared).
     * Accepts the float32 encoding used throughout Pml\Classic classifiers.
     *
     * @param Tensor $y_true  True labels   [n_samples]
     * @param Tensor $y_pred  Predicted labels [n_samples]
     * @return float          Fraction correct ∈ [0, 1]
     */
    public static function accuracy_score(Tensor $y_true, Tensor $y_pred): float
    {
        $n = $y_true->size;
        if ($y_pred->size !== $n) {
            throw new \InvalidArgumentException(
                "accuracy_score: y_true ({$n}) and y_pred ({$y_pred->size}) must be the same length."
            );
        }

        $correct = 0;
        for ($i = 0; $i < $n; $i++) {
            // Round to nearest int before comparing — tolerates floating point
            // representations of integer class labels (0.0, 1.0, 2.0 …).
            if ((int) round((float) $y_true->buffer[$i])
                    === (int) round((float) $y_pred->buffer[$i])) {
                $correct++;
            }
        }

        return $correct / $n;
    }

    // ── Regression ─────────────────────────────────────────────────────────

    /**
     * Mean Absolute Error: MAE = (1/n) · Σ|y_true − y_pred|
     *
     * The average magnitude of prediction errors, in the same units as the
     * target.  Unlike MSE, MAE weights all errors equally — it is robust to
     * outliers because large residuals do not receive disproportionately high
     * penalties.
     *
     * BLAS note: cblas_sasum computes Σ|x| over a vector, but it operates on
     * a single buffer.  We need Σ|y_true[i] − y_pred[i]|, which requires the
     * residual to be materialised first.  We therefore:
     *   1. Copy y_true into a diff buffer      — cblas_scopy
     *   2. diff -= y_pred                      — cblas_saxpy(n, −1, y_pred, diff)
     *   3. MAE = cblas_sasum(diff) / n         — single BLAS-1 call on residual
     *
     * @param Tensor $y_true  Ground-truth targets  [n_samples]
     * @param Tensor $y_pred  Predicted values       [n_samples]
     * @return float          MAE ≥ 0, in target units
     * @throws \InvalidArgumentException if array sizes differ.
     */
    public static function mean_absolute_error(Tensor $y_true, Tensor $y_pred): float
    {
        $n = $y_true->size;
        if ($y_pred->size !== $n) {
            throw new \InvalidArgumentException(
                "mean_absolute_error: y_true ({$n}) and y_pred ({$y_pred->size}) must be the same length."
            );
        }

        $blas = BlasEngine::get()->ffi;

        // diff = y_true − y_pred
        $diff = new Tensor([$n]);
        $blas->cblas_scopy($n, $y_true->buffer, 1, $diff->buffer, 1);
        $blas->cblas_saxpy($n, -1.0, $y_pred->buffer, 1, $diff->buffer, 1);

        // sasum(diff) = Σ|diff[i]|  — signed residuals, so absolute value is correct
        return (float) $blas->cblas_sasum($n, $diff->buffer, 1) / $n;
    }

    /**
     * Mean Squared Error: MSE = (1/n) · ||y_true − y_pred||²
     *
     * BLAS steps:
     *   1. diff = copy(y_true)              — cblas_scopy
     *   2. diff -= y_pred                   — cblas_saxpy(n, −1, y_pred, diff)
     *   3. MSE  = sdot(diff, diff) / n      — cblas_sdot
     *
     * @param Tensor $y_true  Ground-truth targets  [n_samples]
     * @param Tensor $y_pred  Predicted values       [n_samples]
     * @return float          MSE ≥ 0
     */
    public static function mean_squared_error(Tensor $y_true, Tensor $y_pred): float
    {
        $n = $y_true->size;
        if ($y_pred->size !== $n) {
            throw new \InvalidArgumentException(
                "mean_squared_error: y_true ({$n}) and y_pred ({$y_pred->size}) must be the same length."
            );
        }

        $blas = BlasEngine::get()->ffi;

        // diff = y_true − y_pred
        $diff = new Tensor([$n]);
        $blas->cblas_scopy($n, $y_true->buffer, 1, $diff->buffer, 1);
        $blas->cblas_saxpy($n, -1.0, $y_pred->buffer, 1, $diff->buffer, 1);

        // MSE = ||diff||² / n
        return (float) $blas->cblas_sdot($n, $diff->buffer, 1, $diff->buffer, 1) / $n;
    }

    /**
     * R² (coefficient of determination): R² = 1 − SS_res / SS_tot
     *
     * SS_res = ||y_true − y_pred||²          (sum of squared residuals)
     * SS_tot = ||y_true − ȳ||²              (total sum of squares)
     *
     * Returns 1.0 when SS_tot = 0 (constant target vector — consistent
     * with sklearn's behaviour).
     *
     * BLAS steps for SS_tot:
     *   1. ȳ = mean(y_true)               — PHP loop (unavoidable for mean)
     *   2. tot  = copy(y_true)            — cblas_scopy
     *   3. tot -= ȳ · ones               — cblas_saxpy(n, −ȳ, ones, tot)
     *   4. SS_tot = sdot(tot, tot)        — cblas_sdot
     *
     * @param Tensor $y_true  Ground-truth targets  [n_samples]
     * @param Tensor $y_pred  Predicted values       [n_samples]
     * @return float          R² ∈ (−∞, 1]
     */
    public static function r2_score(Tensor $y_true, Tensor $y_pred): float
    {
        $n = $y_true->size;
        if ($y_pred->size !== $n) {
            throw new \InvalidArgumentException(
                "r2_score: y_true ({$n}) and y_pred ({$y_pred->size}) must be the same length."
            );
        }

        $blas = BlasEngine::get()->ffi;

        // ── SS_res = ||y_true − y_pred||² ─────────────────────────────────
        $res = new Tensor([$n]);
        $blas->cblas_scopy($n, $y_true->buffer, 1, $res->buffer, 1);
        $blas->cblas_saxpy($n, -1.0, $y_pred->buffer, 1, $res->buffer, 1);
        $ss_res = (float) $blas->cblas_sdot($n, $res->buffer, 1, $res->buffer, 1);

        // ── SS_tot = ||y_true − ȳ||² ──────────────────────────────────────
        //
        // PHP loop to compute ȳ — BLAS sasum sums absolute values (not signed),
        // and there is no signed-sum BLAS primitive.  This loop is O(n) with
        // no allocation and is dominated by the BLAS calls.
        $yMean = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $yMean += (float) $y_true->buffer[$i];
        }
        $yMean /= $n;

        // tot = y_true − ȳ   via saxpy: tot[i] += −ȳ · ones[i] = −ȳ
        $tot  = new Tensor([$n]);
        $ones = Tensor::ones([$n]);
        $blas->cblas_scopy($n, $y_true->buffer, 1, $tot->buffer, 1);
        $blas->cblas_saxpy($n, -$yMean, $ones->buffer, 1, $tot->buffer, 1);
        $ss_tot = (float) $blas->cblas_sdot($n, $tot->buffer, 1, $tot->buffer, 1);

        // Constant target vector: all true values are identical → SS_tot = 0.
        // R² is undefined in this case; sklearn returns 0.0 (not 1.0) because
        // the model cannot be said to explain variance that does not exist.
        if ($ss_tot === 0.0) {
            return 0.0;
        }

        return 1.0 - $ss_res / $ss_tot;
    }
}
