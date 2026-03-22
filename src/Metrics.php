<?php

declare(strict_types=1);

namespace Pml;

/**
 * Metrics: Model evaluation metrics for classification, regression, and language modelling.
 *
 * Conventions:
 *   - $logits : Tensor [N, C] — raw (pre-softmax) model output.
 *   - $targets : int[]  [N]   — ground-truth class indices (0-based).
 *   - $yPred  : int[]  [N]   — predicted class indices (argmax already applied).
 *   - $pred / $target : Tensor [*] — matching shapes, for regression metrics.
 */
final class Metrics
{
    // ── Classification ────────────────────────────────────────────────────

    /**
     * Top-1 accuracy: fraction of samples where argmax(logits) == target.
     *
     * @param  Tensor  $logits  [N, C]
     * @param  int[]   $targets [N]
     */
    public static function accuracy(Tensor $logits, array $targets): float
    {
        $n       = $logits->shape[0];
        $correct = 0;
        foreach (self::argmax($logits) as $i => $pred) {
            if ($pred === $targets[$i]) $correct++;
        }
        return $correct / $n;
    }

    /**
     * Top-k accuracy: fraction of samples where the true class is among the
     * top-k predicted classes.
     *
     * @param  Tensor  $logits  [N, C]
     * @param  int[]   $targets [N]
     */
    public static function topKAccuracy(Tensor $logits, array $targets, int $k): float
    {
        $n    = $logits->shape[0];
        $C    = $logits->shape[1];
        $k    = min($k, $C);
        $hits = 0;

        for ($i = 0; $i < $n; $i++) {
            $row = [];
            for ($j = 0; $j < $C; $j++) {
                $row[$j] = (float) $logits->buffer[$i * $C + $j];
            }
            arsort($row);
            $topK = array_slice(array_keys($row), 0, $k);
            if (in_array($targets[$i], $topK, true)) $hits++;
        }

        return $hits / $n;
    }

    /**
     * Confusion matrix. Returns a Tensor [C, C] where entry [i, j] is the
     * number of samples with true class i predicted as class j.
     *
     * @param  int[]  $yPred      [N] predicted class indices
     * @param  int[]  $yTrue      [N] ground-truth class indices
     * @param  int    $numClasses C
     */
    public static function confusionMatrix(array $yPred, array $yTrue, int $numClasses): Tensor
    {
        $cm = Tensor::zeros([$numClasses, $numClasses]);
        foreach ($yTrue as $i => $true) {
            $cm->buffer[$true * $numClasses + $yPred[$i]] += 1.0;
        }
        return $cm;
    }

    /**
     * Per-class and averaged Precision, Recall, and F1 score.
     *
     * Returns an associative array:
     *   'precision' => float[]  per-class precision
     *   'recall'    => float[]  per-class recall
     *   'f1'        => float[]  per-class F1
     *   'macro_precision' => float
     *   'macro_recall'    => float
     *   'macro_f1'        => float
     *   'micro_precision' => float  (== micro_recall == micro_f1 for multi-class)
     *
     * @param  int[]  $yPred      [N]
     * @param  int[]  $yTrue      [N]
     * @param  int    $numClasses C
     */
    public static function precisionRecallF1(array $yPred, array $yTrue, int $numClasses): array
    {
        $tp = array_fill(0, $numClasses, 0);
        $fp = array_fill(0, $numClasses, 0);
        $fn = array_fill(0, $numClasses, 0);

        foreach ($yTrue as $i => $true) {
            $pred = $yPred[$i];
            if ($pred === $true) {
                $tp[$true]++;
            } else {
                $fp[$pred]++;
                $fn[$true]++;
            }
        }

        $precision = [];
        $recall    = [];
        $f1        = [];

        for ($c = 0; $c < $numClasses; $c++) {
            $p           = $tp[$c] + $fp[$c] > 0 ? $tp[$c] / ($tp[$c] + $fp[$c]) : 0.0;
            $r           = $tp[$c] + $fn[$c] > 0 ? $tp[$c] / ($tp[$c] + $fn[$c]) : 0.0;
            $precision[] = $p;
            $recall[]    = $r;
            $f1[]        = ($p + $r) > 0.0 ? 2 * $p * $r / ($p + $r) : 0.0;
        }

        $totalTP = array_sum($tp);
        $totalFP = array_sum($fp);
        $totalFN = array_sum($fn);
        $microP  = $totalTP + $totalFP > 0 ? $totalTP / ($totalTP + $totalFP) : 0.0;
        $microR  = $totalTP + $totalFN > 0 ? $totalTP / ($totalTP + $totalFN) : 0.0;
        $microF1 = ($microP + $microR) > 0.0 ? 2 * $microP * $microR / ($microP + $microR) : 0.0;

        return [
            'precision'       => $precision,
            'recall'          => $recall,
            'f1'              => $f1,
            'macro_precision' => array_sum($precision) / $numClasses,
            'macro_recall'    => array_sum($recall)    / $numClasses,
            'macro_f1'        => array_sum($f1)        / $numClasses,
            'micro_precision' => $microP,
            'micro_recall'    => $microR,
            'micro_f1'        => $microF1,
        ];
    }

    // ── Regression ────────────────────────────────────────────────────────

    /**
     * Mean Absolute Error: mean(|pred - target|)
     */
    public static function mae(Tensor $pred, Tensor $target): float
    {
        if ($pred->size !== $target->size) {
            throw new \InvalidArgumentException('MAE: size mismatch.');
        }
        $sum = 0.0;
        for ($i = 0; $i < $pred->size; $i++) {
            $sum += abs((float) $pred->buffer[$i] - (float) $target->buffer[$i]);
        }
        return $sum / $pred->size;
    }

    /**
     * Root Mean Squared Error: sqrt(mean((pred - target)²))
     */
    public static function rmse(Tensor $pred, Tensor $target): float
    {
        return sqrt(Ops::mseLoss($pred, $target));
    }

    /**
     * Coefficient of determination R²: 1 - SS_res / SS_tot
     * Returns 1.0 for a perfect fit, ≤ 0 for a model no better than the mean.
     */
    public static function r2Score(Tensor $pred, Tensor $target): float
    {
        if ($pred->size !== $target->size) {
            throw new \InvalidArgumentException('R²: size mismatch.');
        }
        $n      = $pred->size;
        $mean   = 0.0;
        for ($i = 0; $i < $n; $i++) $mean += (float) $target->buffer[$i];
        $mean /= $n;

        $ssTot = 0.0;
        $ssRes = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $t      = (float) $target->buffer[$i];
            $p      = (float) $pred->buffer[$i];
            $ssTot += ($t - $mean) ** 2;
            $ssRes += ($t - $p)    ** 2;
        }

        return $ssTot < 1e-12 ? 1.0 : 1.0 - $ssRes / $ssTot;
    }

    // ── Language Model ────────────────────────────────────────────────────

    /**
     * Perplexity: exp(cross_entropy_loss).
     * Lower is better. A uniform distribution over C classes has perplexity C.
     *
     * @param  Tensor  $logits  [N, C]
     * @param  int[]   $targets [N]
     */
    public static function perplexity(Tensor $logits, array $targets): float
    {
        return exp(Ops::crossEntropyLoss($logits, $targets));
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /**
     * Argmax over each row of a 2D tensor [N, C]. Returns int[N].
     */
    public static function argmax(Tensor $logits): array
    {
        $n    = $logits->shape[0];
        $C    = $logits->shape[1];
        $preds = [];

        for ($i = 0; $i < $n; $i++) {
            $best  = -1;
            $bestV = -INF;
            $off   = $i * $C;
            for ($j = 0; $j < $C; $j++) {
                $v = (float) $logits->buffer[$off + $j];
                if ($v > $bestV) { $bestV = $v; $best = $j; }
            }
            $preds[] = $best;
        }

        return $preds;
    }
}
