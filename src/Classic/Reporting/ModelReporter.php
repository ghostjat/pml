<?php

declare(strict_types=1);

namespace Pml\Classic\Reporting;

use Pml\Tensor;
use Pml\Classic\Predictor;
use Pml\Classic\Metrics\Metrics;

// ═══════════════════════════════════════════════════════════════════════════
//  ModelReporter — Automated Model Governance and Reporting
//
//  Evaluates a fitted Predictor against held-out test data and serialises
//  a rich, human-readable JSON report containing:
//    • Dataset metadata (n_samples, n_features, feature names)
//    • Task-appropriate metrics (classification or regression)
//    • Automated data science insights derived from metric relationships
//
//  ── Usage ─────────────────────────────────────────────────────────────────
//
//    $json = ModelReporter::generate($pipe, $Xtest, $ytest, $featureNames);
//    file_put_contents('report.json', $json);
//
//  ── Task Detection ────────────────────────────────────────────────────────
//
//  When $task = 'auto' (default), the reporter inspects every value in
//  $y_test.  If all values satisfy round($v) == $v (i.e. every label is an
//  integer), the task is classified as 'classification'.  Continuous floats
//  trigger 'regression'.  This heuristic covers the typical Pml\Classic use
//  cases.  Override with $task = 'classification' or 'regression' if the
//  auto-detection is ambiguous (e.g. an integer-valued regression target).
//
//  ── JSON Structure ────────────────────────────────────────────────────────
//
//  {
//    "model":         "XGBClassifier",
//    "task":          "classification",
//    "generated_at":  "2026-03-23T14:05:00+00:00",
//    "dataset": { "n_samples": 179, "n_features": 10, "feature_names": [...] },
//    "metrics": {                          ← task-specific block
//      "accuracy": 0.8212,
//      "precision": 0.8108,
//      "recall": 0.7500,
//      "f1_score": 0.7792,
//      "specificity": 0.8750,
//      "confusion_matrix": { "TP": 60, "TN": 91, "FP": 14, "FN": 20 },
//      "baseline": { "majority_class_accuracy": 0.6145, "lift_pp": 20.67 }
//    },
//    "insights": [ "...", "..." ]
//  }
//
//  Regression variant replaces "metrics" with:
//    { "mae": ..., "mse": ..., "rmse": ..., "r2": ... }
// ═══════════════════════════════════════════════════════════════════════════

final class ModelReporter
{
    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Evaluate a fitted model and return a JSON report string.
     *
     * @param Predictor  $model         A fitted Pml\Classic\Predictor (Pipeline, classifier, regressor…)
     * @param Tensor     $X_test        Feature matrix  [n_samples, n_features]
     * @param Tensor     $y_test        Ground-truth labels / targets  [n_samples]
     * @param string[]   $feature_names Column names parallel to $X_test's columns (optional)
     * @param string     $task          'auto' | 'classification' | 'regression'
     *
     * @return string  JSON-encoded report (UTF-8, pretty-printed)
     *
     * @throws \InvalidArgumentException If $task is not a valid option.
     */
    public static function generate(
        Predictor $model,
        Tensor    $X_test,
        Tensor    $y_test,
        array     $feature_names = [],
        string    $task          = 'auto',
    ): string {
        if (!in_array($task, ['auto', 'classification', 'regression'], true)) {
            throw new \InvalidArgumentException(
                "ModelReporter: task must be 'auto', 'classification', or 'regression', got '{$task}'."
            );
        }

        // ── Resolve task ──────────────────────────────────────────────────
        $resolvedTask = ($task === 'auto')
            ? self::detectTask($y_test)
            : $task;

        // ── Run predictions ───────────────────────────────────────────────
        $y_pred = $model->predict($X_test);

        // ── Build report skeleton ─────────────────────────────────────────
        $nFeatures = (count($X_test->shape) >= 2) ? $X_test->shape[1] : $X_test->size;

        $report = [
            'model'        => self::shortClassName($model),
            'task'         => $resolvedTask,
            'generated_at' => (new \DateTimeImmutable())->format(\DateTimeInterface::ATOM),
            'dataset'      => [
                'n_samples'     => $X_test->shape[0],
                'n_features'    => $nFeatures,
                'feature_names' => $feature_names,
            ],
        ];

        // ── Dispatch to task-specific reporter ────────────────────────────
        if ($resolvedTask === 'classification') {
            [$metrics, $insights] = self::generateClassificationReport($y_test, $y_pred);
        } else {
            [$metrics, $insights] = self::generateRegressionReport($y_test, $y_pred);
        }

        $report['metrics']  = $metrics;
        $report['insights'] = $insights;

        return (string) json_encode($report, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    }

    // ── Task Detection ────────────────────────────────────────────────────

    /**
     * Inspect y values — all integer-valued floats → 'classification', else 'regression'.
     */
    private static function detectTask(Tensor $y): string
    {
        $n = $y->size;
        for ($i = 0; $i < $n; $i++) {
            $v = (float) $y->buffer[$i];
            if (round($v) !== $v) {
                return 'regression';
            }
        }
        return 'classification';
    }

    // ── Classification ────────────────────────────────────────────────────

    /**
     * Compute binary classification metrics and automated insights.
     *
     * Assumes binary labels: positive class = 1, negative class = 0.
     *
     * @return array{0: array<string, mixed>, 1: string[]}
     */
    private static function generateClassificationReport(Tensor $y_true, Tensor $y_pred): array
    {
        $n = $y_true->size;

        // ── Confusion matrix ───────────────────────────────────────────
        $TP = $TN = $FP = $FN = 0;
        for ($i = 0; $i < $n; $i++) {
            $pred  = (int) round((float) $y_pred->buffer[$i]);
            $truth = (int) round((float) $y_true->buffer[$i]);

            match (true) {
                $pred === 1 && $truth === 1 => $TP++,
                $pred === 0 && $truth === 0 => $TN++,
                $pred === 1 && $truth === 0 => $FP++,
                default                     => $FN++,
            };
        }

        // ── Derived metrics ────────────────────────────────────────────
        $accuracy    = ($TP + $TN) / $n;
        $precision   = ($TP + $FP) > 0 ? $TP / ($TP + $FP) : 0.0;
        $recall      = ($TP + $FN) > 0 ? $TP / ($TP + $FN) : 0.0;
        $specificity = ($TN + $FP) > 0 ? $TN / ($TN + $FP) : 0.0;
        $f1          = $precision + $recall > 0
            ? 2.0 * $precision * $recall / ($precision + $recall)
            : 0.0;

        $majorityBaseline = ($TN + $FN) / $n;
        $liftPp           = ($accuracy - $majorityBaseline) * 100.0;

        $metrics = [
            'accuracy'    => round($accuracy,    4),
            'precision'   => round($precision,   4),
            'recall'      => round($recall,      4),
            'f1_score'    => round($f1,          4),
            'specificity' => round($specificity, 4),
            'confusion_matrix' => [
                'TP' => $TP,
                'TN' => $TN,
                'FP' => $FP,
                'FN' => $FN,
            ],
            'baseline' => [
                'majority_class_accuracy' => round($majorityBaseline, 4),
                'lift_pp'                 => round($liftPp, 2),
            ],
        ];

        // ── Automated insights ─────────────────────────────────────────
        $insights = self::classificationInsights(
            $accuracy, $precision, $recall, $f1, $specificity,
            $TP, $TN, $FP, $FN, $majorityBaseline
        );

        return [$metrics, $insights];
    }

    /**
     * Generate human-readable classification insights from metric relationships.
     *
     * @return string[]
     */
    private static function classificationInsights(
        float $accuracy,
        float $precision,
        float $recall,
        float $f1,
        float $specificity,
        int   $TP, int $TN, int $FP, int $FN,
        float $baseline,
    ): array {
        $insights = [];

        // ── Overall quality ────────────────────────────────────────────
        if ($accuracy >= 0.90) {
            $insights[] = 'Excellent accuracy (≥90%). The model generalises well to unseen data.';
        } elseif ($accuracy >= 0.80) {
            $insights[] = 'Good accuracy (80–90%). The model captures most patterns in the data.';
        } elseif ($accuracy >= 0.70) {
            $insights[] = 'Moderate accuracy (70–80%). There is meaningful room for improvement — consider additional feature engineering or hyperparameter tuning.';
        } else {
            $insights[] = 'Low accuracy (<70%). The model is underperforming. Investigate class distributions, feature quality, and model capacity.';
        }

        // ── Class imbalance trap ───────────────────────────────────────
        if ($accuracy - $baseline > 0.30) {
            $insights[] = sprintf(
                'Strong lift over the majority-class baseline (+%.1f pp). The model is genuinely discriminative, not just exploiting class imbalance.',
                ($accuracy - $baseline) * 100
            );
        } elseif (($accuracy - $baseline) < 0.05 && $f1 < 0.60) {
            $insights[] = 'Warning: High accuracy driven by class imbalance. The accuracy score is close to the majority-class baseline and F1 is low — the model struggles significantly with the minority class. Consider class-weighted loss, oversampling (SMOTE), or a lower classification threshold.';
        }

        // ── Conservative / pessimistic model ──────────────────────────
        if ($recall < 0.70 && $precision > 0.80) {
            $insights[] = 'The model is overly conservative (pessimistic). It prioritises exactness over finding all positive cases, resulting in a high False Negative rate. If missing true positives is costly (e.g. medical screening, fraud detection), lower the decision threshold to improve recall.';
        }

        // ── Aggressive / optimistic model ─────────────────────────────
        if ($precision < 0.60 && $recall > 0.85) {
            $insights[] = 'The model is overly aggressive (optimistic). It flags too many negatives as positive (high False Positive rate). If false alarms are costly, raise the decision threshold to improve precision.';
        }

        // ── Balanced precision/recall ──────────────────────────────────
        if (abs($precision - $recall) <= 0.05 && $f1 >= 0.75) {
            $insights[] = sprintf(
                'Precision (%.2f) and Recall (%.2f) are well-balanced (Δ ≤ 0.05), yielding a strong F1-score of %.2f. No threshold adjustment is needed for the current objective.',
                $precision, $recall, $f1
            );
        }

        // ── High specificity but low recall ───────────────────────────
        if ($specificity > 0.90 && $recall < 0.65) {
            $insights[] = 'The model is excellent at identifying negatives (specificity > 90%) but misses a significant fraction of positives. This profile suits contexts where false alarms are very costly, but may be inappropriate when missing a positive case is dangerous.';
        }

        // ── FP vs FN bias ──────────────────────────────────────────────
        if ($FP > $FN * 1.5) {
            $insights[] = sprintf(
                'Directional bias — OPTIMISTIC: the model generates %d False Positives vs %d False Negatives. It over-predicts the positive class. Tune the threshold upward to reduce FP.',
                $FP, $FN
            );
        } elseif ($FN > $FP * 1.5) {
            $insights[] = sprintf(
                'Directional bias — PESSIMISTIC: the model generates %d False Negatives vs %d False Positives. It under-predicts the positive class. Tune the threshold downward to reduce FN.',
                $FN, $FP
            );
        }

        return $insights;
    }

    // ── Regression ────────────────────────────────────────────────────────

    /**
     * Compute regression metrics and automated insights.
     *
     * @return array{0: array<string, mixed>, 1: string[]}
     */
    private static function generateRegressionReport(Tensor $y_true, Tensor $y_pred): array
    {
        $n = $y_true->size;

        // ── MAE — pure PHP loop (no BLAS signed-sum) ──────────────────
        $sumAbs = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $sumAbs += abs((float) $y_true->buffer[$i] - (float) $y_pred->buffer[$i]);
        }
        $mae = $sumAbs / $n;

        // ── MSE and R² — delegate to Metrics (BLAS-accelerated) ───────
        $mse = Metrics::mean_squared_error($y_true, $y_pred);
        $r2  = Metrics::r2_score($y_true, $y_pred);
        $rmse = sqrt($mse);

        // ── Target mean and std for scale context ─────────────────────
        $ySum = 0.0;
        for ($i = 0; $i < $n; $i++) {
            $ySum += (float) $y_true->buffer[$i];
        }
        $yMean = $ySum / $n;

        $metrics = [
            'mae'  => round($mae,  4),
            'mse'  => round($mse,  4),
            'rmse' => round($rmse, 4),
            'r2'   => round($r2,   4),
        ];

        $insights = self::regressionInsights($mae, $rmse, $r2, $yMean);

        return [$metrics, $insights];
    }

    /**
     * Generate human-readable regression insights from metric relationships.
     *
     * @return string[]
     */
    private static function regressionInsights(
        float $mae,
        float $rmse,
        float $r2,
        float $yMean,
    ): array {
        $insights = [];

        // ── R² quality tiers ───────────────────────────────────────────
        if ($r2 >= 0.90) {
            $insights[] = sprintf(
                'Excellent R² (%.2f). The model explains %.1f%% of the variance in the target — a strong fit.',
                $r2, $r2 * 100
            );
        } elseif ($r2 >= 0.70) {
            $insights[] = sprintf(
                'Good R² (%.2f). The model explains %.1f%% of the variance. Residuals may contain exploitable signal — try polynomial features or interaction terms.',
                $r2, $r2 * 100
            );
        } elseif ($r2 >= 0.50) {
            $insights[] = sprintf(
                'Moderate R² (%.2f). The model explains %.1f%% of the variance. Consider adding non-linear features, a richer feature set, or switching to a tree-based ensemble.',
                $r2, $r2 * 100
            );
        } else {
            $insights[] = sprintf(
                'Low R² (%.2f). The model explains less than 50%% of the variance in the target. The current feature set may be insufficient. Consider adding domain-specific features, non-linear transformations, or a more expressive model (e.g. gradient boosting).',
                $r2
            );
        }

        // ── Negative R² ───────────────────────────────────────────────
        if ($r2 < 0.0) {
            $insights[] = 'R² is negative — the model performs worse than a constant mean predictor. This usually indicates a data leakage issue, incorrect target encoding, or severe train/test distribution mismatch. Verify the preprocessing pipeline was fitted on training data only.';
        }

        // ── RMSE vs MAE — skewness of errors ──────────────────────────
        if ($rmse > $mae * 1.5) {
            $insights[] = sprintf(
                'RMSE (%.4f) is significantly larger than MAE (%.4f) — a RMSE/MAE ratio of %.2f. This indicates the presence of large individual errors (outliers in the residuals). Investigate high-error samples or consider a robust loss function (e.g. Huber, quantile regression).',
                $rmse, $mae, ($mae > 0 ? $rmse / $mae : 0)
            );
        } else {
            $insights[] = sprintf(
                'RMSE (%.4f) is close to MAE (%.4f), indicating that errors are uniformly distributed with no dominant outliers. The model\'s error profile is consistent across the test set.',
                $rmse, $mae
            );
        }

        // ── Error relative to target mean ─────────────────────────────
        if ($yMean !== 0.0) {
            $mapeProxy = ($mae / abs($yMean)) * 100;
            if ($mapeProxy < 10) {
                $insights[] = sprintf(
                    'MAE is %.1f%% of the target mean — errors are small relative to the scale of the target.',
                    $mapeProxy
                );
            } elseif ($mapeProxy < 25) {
                $insights[] = sprintf(
                    'MAE is %.1f%% of the target mean — moderate relative error. Consider whether this level of imprecision is acceptable for the application.',
                    $mapeProxy
                );
            } else {
                $insights[] = sprintf(
                    'MAE is %.1f%% of the target mean — errors are large relative to the target scale. The model may benefit from log-transforming a skewed target before training.',
                    $mapeProxy
                );
            }
        }

        return $insights;
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Return only the short (unqualified) class name of an object.
     */
    private static function shortClassName(object $obj): string
    {
        $full  = get_class($obj);
        $parts = explode('\\', $full);
        return end($parts);
    }
}
