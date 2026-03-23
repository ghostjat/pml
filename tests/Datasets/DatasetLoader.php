<?php

declare(strict_types=1);

namespace Pml\Tests\Datasets;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  DatasetLoader — Self-contained dataset provider for Pml test suites
//
//  All datasets are either hardcoded inline (Iris) or procedurally generated
//  (synthetic regression / blobs / classification).  No HTTP requests, no
//  external files.
//
//  ── Iris Dataset ─────────────────────────────────────────────────────────
//
//  The Fisher Iris dataset (1936): 150 samples, 4 features, 3 classes.
//
//  Features (all in cm):
//    col 0 — sepal length
//    col 1 — sepal width
//    col 2 — petal length
//    col 3 — petal width
//
//  Classes:
//    0 — Iris setosa     (rows   0– 49)
//    1 — Iris versicolor (rows  50– 99)
//    2 — Iris virginica  (rows 100–149)
//
//  Setosa is linearly separable from the other two on petal features.
//  Versicolor and virginica overlap slightly on sepal features.
//  A Random Forest or Logistic Regression achieves ≥ 96% on a held-out set.
//
//  ── Synthetic Regression ─────────────────────────────────────────────────
//
//  y = 3·x₁ − 2·x₂ + 5 + ε,   ε ~ N(0, noise_std²)
//
//  x₁, x₂ ~ Uniform(−3, 3)  independently.
//  With n=500 and noise_std=0.5, a properly-regularised linear model recovers
//  [3, −2] with less than 0.05 absolute error on each coefficient.
//  R² is typically > 0.98 on the same training data.
//
//  ── make_blobs ───────────────────────────────────────────────────────────
//
//  Isotropic Gaussian clusters.  Cluster k is centred at
//  (k·sep, k·sep, ...) in all $n_features dimensions.  With sep=5.0 and
//  cluster_std=0.9 the clusters are ~5/0.9 ≈ 5.6 σ apart — well beyond the
//  decision boundary for any distance-based clustering algorithm.
//
//  ── make_imputation_data ─────────────────────────────────────────────────
//
//  Small 2D float matrix with deterministic NaN injection (every 10th cell).
//  Returns the per-column means of the non-NaN values so tests can optionally
//  verify that SimpleImputer::fit_transform() recovered the correct fill value.
//
//  ── make_classification ──────────────────────────────────────────────────
//
//  Balanced Gaussian blobs, one per class.  Class k centred at k·5 in all
//  features.  With cluster_std=0.8 the margin is >6 σ — easily linearly
//  separable — so binary and multiclass classifiers should achieve >95 %.
// ═══════════════════════════════════════════════════════════════════════════

final class DatasetLoader
{
    // ── Iris (hardcoded CSV) ───────────────────────────────────────────────

    /**
     * Load the Fisher Iris dataset.
     *
     * Returns ['X' => Tensor[150, 4], 'y' => Tensor[150]].
     * X is float32, y is float32 with integer class labels 0.0/1.0/2.0.
     */
    public static function iris(): array
    {
        $rows = self::parseIrisCsv(self::IRIS_CSV);

        $n = count($rows);         // 150
        $d = 4;

        $X = new Tensor([$n, $d]);
        $y = new Tensor([$n]);

        foreach ($rows as $i => $row) {
            for ($j = 0; $j < $d; $j++) {
                $X->buffer[$i * $d + $j] = (float)$row[$j];
            }
            $y->buffer[$i] = (float)$row[$d];   // class label (0, 1, or 2)
        }

        return ['X' => $X, 'y' => $y];
    }

    /**
     * Parse the inline CSV string into a 2D float array.
     *
     * @return float[][]  Each inner array is [f0, f1, f2, f3, class_label].
     */
    private static function parseIrisCsv(string $csv): array
    {
        $rows = [];
        foreach (explode("\n", trim($csv)) as $line) {
            $line = trim($line);
            if ($line === '' || str_starts_with($line, '#')) {
                continue;
            }
            $cols = explode(',', $line);
            if (count($cols) < 5) {
                continue;
            }
            $rows[] = array_map('floatval', $cols);
        }
        return $rows;
    }

    // ── Synthetic regression ───────────────────────────────────────────────

    /**
     * Generate the synthetic regression dataset  y = 3x₁ − 2x₂ + 5 + ε.
     *
     * Uses a simple LCG (linear congruential generator) so results are fully
     * reproducible on any platform without needing mt_srand state.
     *
     * Returns ['X' => Tensor[n, 2], 'y' => Tensor[n]].
     *
     * @param int   $n         Number of samples (default 500).
     * @param float $noise_std Standard deviation of the additive Gaussian noise.
     * @param int   $seed      LCG seed for reproducibility.
     */
    public static function synthetic_regression(
        int   $n         = 500,
        float $noise_std = 0.5,
        int   $seed      = 42,
    ): array {
        $X = new Tensor([$n, 2]);
        $y = new Tensor([$n]);

        // ── LCG parameters (Knuth, The Art of Computer Programming §3.3.4)
        // Generates uniform samples in [0,1).
        // Two independent streams (seed and seed+1) for x1 and x2.
        // A third stream (seed+2) for Gaussian noise via Box-Muller transform.
        $lcgM = 4294967296;   // 2^32
        $lcgA = 1664525;
        $lcgC = 1013904223;

        $state1 = $seed & 0xFFFFFFFF;
        $state2 = ($seed + 1) & 0xFFFFFFFF;
        $state3 = ($seed + 2) & 0xFFFFFFFF;
        $state4 = ($seed + 3) & 0xFFFFFFFF;

        $lcg = static function(int &$state) use ($lcgA, $lcgC, $lcgM): float {
            $state = (($lcgA * $state + $lcgC) & 0xFFFFFFFF);
            return $state / $lcgM;
        };

        for ($i = 0; $i < $n; $i++) {
            // x1, x2 ~ Uniform(−3, 3)
            $x1 = $lcg($state1) * 6.0 - 3.0;
            $x2 = $lcg($state2) * 6.0 - 3.0;

            // Box-Muller transform: U1, U2 ~ Uniform(0,1) → Z ~ N(0,1)
            // Z = sqrt(-2·ln(U1)) · cos(2π·U2)
            // We use two streams so x-generation and noise-generation are independent.
            $u1 = max($lcg($state3), 1e-10);   // avoid ln(0)
            $u2 = $lcg($state4);
            $z  = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
            $eps = $z * $noise_std;

            $X->buffer[$i * 2]     = (float)$x1;
            $X->buffer[$i * 2 + 1] = (float)$x2;

            // Ground-truth: y = 3·x₁ − 2·x₂ + 5 + ε
            $y->buffer[$i] = (float)(3.0 * $x1 - 2.0 * $x2 + 5.0 + $eps);
        }

        return ['X' => $X, 'y' => $y];
    }

    // ── make_blobs ────────────────────────────────────────────────────────

    /**
     * Generate isotropic Gaussian blobs for clustering tests.
     *
     * Cluster k is centred at (k·$sep, k·$sep, …) in every feature dimension.
     * Samples are allocated in contiguous blocks per cluster (no shuffle) —
     * KMeans does not require shuffled input.
     *
     * Uses the same seeded LCG + Box-Muller strategy as synthetic_regression()
     * so results are platform-independent.
     *
     * @return array{X: Tensor, y: Tensor}  X is [n_samples, n_features], y is [n_samples].
     */
    public static function make_blobs(
        int   $n_samples   = 100,
        int   $centers     = 3,
        int   $n_features  = 2,
        float $cluster_std = 0.9,
        int   $seed        = 42,
    ): array {
        $X = new Tensor([$n_samples, $n_features]);
        $y = new Tensor([$n_samples]);

        // ── LCG (same constants as synthetic_regression) ─────────────────
        $lcgM = 4294967296;   // 2^32
        $lcgA = 1664525;
        $lcgC = 1013904223;

        // Two independent streams — one for Box-Muller u1, one for u2
        $state1 = ($seed)     & 0xFFFFFFFF;
        $state2 = ($seed + 1) & 0xFFFFFFFF;

        $lcg = static function(int &$s) use ($lcgA, $lcgC, $lcgM): float {
            $s = (($lcgA * $s + $lcgC) & 0xFFFFFFFF);
            return $s / $lcgM;
        };

        // Cluster centres lie on the main diagonal: centre_k = k * sep
        // With sep=5.0 and cluster_std=0.9 the clusters are ~5.6 σ apart —
        // well beyond the nearest-neighbour decision boundary.
        $sep = 5.0;

        // Distribute samples evenly; first ($n_samples mod $centers) clusters
        // get one extra sample so the total is exactly $n_samples.
        $base      = intdiv($n_samples, $centers);
        $remainder = $n_samples - $base * $centers;

        $idx = 0;
        for ($k = 0; $k < $centers; $k++) {
            $count  = $base + ($k < $remainder ? 1 : 0);
            $center = (float)($k * $sep);

            for ($i = 0; $i < $count; $i++) {
                for ($j = 0; $j < $n_features; $j++) {
                    // Box-Muller: Z ~ N(0,1)
                    $u1 = max($lcg($state1), 1e-10);
                    $u2 = $lcg($state2);
                    $z  = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
                    $X->buffer[$idx * $n_features + $j] = (float)($center + $z * $cluster_std);
                }
                $y->buffer[$idx] = (float)$k;
                $idx++;
            }
        }

        return ['X' => $X, 'y' => $y];
    }

    // ── make_imputation_data ──────────────────────────────────────────────

    /**
     * Synthetic dataset with deterministic NaN injection for imputer tests.
     *
     * Generates a clean [n_samples, n_features] matrix where feature j has
     * base mean (j+1)·10.0 + small Gaussian noise, then sets every 10th cell
     * (by flat buffer index) to NAN.  Approximately 10% of values become NaN.
     *
     * Returns:
     *   'X'           — Tensor[n_samples, n_features] with NaN holes
     *   'true_means'  — float[] of per-column means computed from non-NaN values
     *                   (i.e. what SimpleImputer(strategy='mean') should recover)
     *   'n_samples'   — int
     *   'n_features'  — int
     */
    public static function make_imputation_data(
        int $n_samples  = 20,
        int $n_features = 4,
        int $seed       = 0,
    ): array {
        $lcgM = 4294967296;
        $lcgA = 1664525;
        $lcgC = 1013904223;

        // Two independent streams for Box-Muller
        $state1 = ($seed + 7)  & 0xFFFFFFFF;
        $state2 = ($seed + 13) & 0xFFFFFFFF;

        $lcg = static function(int &$s) use ($lcgA, $lcgC, $lcgM): float {
            $s = (($lcgA * $s + $lcgC) & 0xFFFFFFFF);
            return $s / $lcgM;
        };

        $X = new Tensor([$n_samples, $n_features]);

        // Fill with clean data: feature j ~ N((j+1)·10, 1²)
        for ($i = 0; $i < $n_samples; $i++) {
            for ($j = 0; $j < $n_features; $j++) {
                $u1 = max($lcg($state1), 1e-10);
                $u2 = $lcg($state2);
                $z  = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
                $X->buffer[$i * $n_features + $j] = (float)(($j + 1) * 10.0 + $z);
            }
        }

        // Inject NaN at every 10th flat-buffer position (deterministic ~10%)
        for ($pos = 0; $pos < $n_samples * $n_features; $pos += 10) {
            $X->buffer[$pos] = NAN;
        }

        // Compute per-column means from the surviving (non-NaN) values —
        // these are the values SimpleImputer(strategy='mean') will fill in.
        $trueMeans = [];
        for ($j = 0; $j < $n_features; $j++) {
            $sum = 0.0;
            $cnt = 0;
            for ($i = 0; $i < $n_samples; $i++) {
                $v = (float)$X->buffer[$i * $n_features + $j];
                if (!is_nan($v)) {
                    $sum += $v;
                    $cnt++;
                }
            }
            $trueMeans[$j] = $cnt > 0 ? $sum / $cnt : 0.0;
        }

        return [
            'X'          => $X,
            'true_means' => $trueMeans,
            'n_samples'  => $n_samples,
            'n_features' => $n_features,
        ];
    }

    // ── make_classification ───────────────────────────────────────────────

    /**
     * Generate a balanced multi-class dataset from well-separated Gaussian blobs.
     *
     * Class k is centred at (k·5, k·5, …) in all $n_features dimensions.
     * With cluster_std=0.8 the nearest-neighbour margin is >6 σ, making the
     * dataset easily linearly separable for any standard classifier.
     *
     * Returns ['X' => Tensor[n_samples, n_features], 'y' => Tensor[n_samples]]
     * with integer class labels 0 … n_classes−1.
     */
    public static function make_classification(
        int $n_samples  = 200,
        int $n_features = 4,
        int $n_classes  = 2,
        int $seed       = 42,
    ): array {
        $X = new Tensor([$n_samples, $n_features]);
        $y = new Tensor([$n_samples]);

        $lcgM = 4294967296;
        $lcgA = 1664525;
        $lcgC = 1013904223;

        $state1 = ($seed + 17) & 0xFFFFFFFF;
        $state2 = ($seed + 31) & 0xFFFFFFFF;

        $lcg = static function(int &$s) use ($lcgA, $lcgC, $lcgM): float {
            $s = (($lcgA * $s + $lcgC) & 0xFFFFFFFF);
            return $s / $lcgM;
        };

        // Class centres: class k sits at k·5 in every feature
        $sep         = 5.0;
        $clusterStd  = 0.8;

        $base      = intdiv($n_samples, $n_classes);
        $remainder = $n_samples - $base * $n_classes;

        $idx = 0;
        for ($k = 0; $k < $n_classes; $k++) {
            $count  = $base + ($k < $remainder ? 1 : 0);
            $center = (float)($k * $sep);

            for ($i = 0; $i < $count; $i++) {
                for ($j = 0; $j < $n_features; $j++) {
                    $u1 = max($lcg($state1), 1e-10);
                    $u2 = $lcg($state2);
                    $z  = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
                    $X->buffer[$idx * $n_features + $j] = (float)($center + $z * $clusterStd);
                }
                $y->buffer[$idx] = (float)$k;
                $idx++;
            }
        }

        return ['X' => $X, 'y' => $y];
    }

    // ── Hardcoded Iris CSV ─────────────────────────────────────────────────
    // Format: sepal_length,sepal_width,petal_length,petal_width,class_label
    // Class labels: 0=setosa, 1=versicolor, 2=virginica
    // Source: Fisher (1936), UCI ML Repository (public domain)

    private const IRIS_CSV = <<<'CSV'
5.1,3.5,1.4,0.2,0
4.9,3.0,1.4,0.2,0
4.7,3.2,1.3,0.2,0
4.6,3.1,1.5,0.2,0
5.0,3.6,1.4,0.2,0
5.4,3.9,1.7,0.4,0
4.6,3.4,1.4,0.3,0
5.0,3.4,1.5,0.2,0
4.4,2.9,1.4,0.2,0
4.9,3.1,1.5,0.1,0
5.4,3.7,1.5,0.2,0
4.8,3.4,1.6,0.2,0
4.8,3.0,1.4,0.1,0
4.3,3.0,1.1,0.1,0
5.8,4.0,1.2,0.2,0
5.7,4.4,1.5,0.4,0
5.4,3.9,1.3,0.4,0
5.1,3.5,1.4,0.3,0
5.7,3.8,1.7,0.3,0
5.1,3.8,1.5,0.3,0
5.4,3.4,1.7,0.2,0
5.1,3.7,1.5,0.4,0
4.6,3.6,1.0,0.2,0
5.1,3.3,1.7,0.5,0
4.8,3.4,1.9,0.2,0
5.0,3.0,1.6,0.2,0
5.0,3.4,1.6,0.4,0
5.2,3.5,1.5,0.2,0
5.2,3.4,1.4,0.2,0
4.7,3.2,1.6,0.2,0
4.8,3.1,1.6,0.2,0
5.4,3.4,1.5,0.4,0
5.2,4.1,1.5,0.1,0
5.5,4.2,1.4,0.2,0
4.9,3.1,1.5,0.2,0
5.0,3.2,1.2,0.2,0
5.5,3.5,1.3,0.2,0
4.9,3.6,1.4,0.1,0
4.4,3.0,1.3,0.2,0
5.1,3.4,1.5,0.2,0
5.0,3.5,1.3,0.3,0
4.5,2.3,1.3,0.3,0
4.4,3.2,1.3,0.2,0
5.0,3.5,1.6,0.6,0
5.1,3.8,1.9,0.4,0
4.8,3.0,1.4,0.3,0
5.1,3.8,1.6,0.2,0
4.6,3.2,1.4,0.2,0
5.3,3.7,1.5,0.2,0
5.0,3.3,1.4,0.2,0
7.0,3.2,4.7,1.4,1
6.4,3.2,4.5,1.5,1
6.9,3.1,4.9,1.5,1
5.5,2.3,4.0,1.3,1
6.5,2.8,4.6,1.5,1
5.7,2.8,4.5,1.3,1
6.3,3.3,4.7,1.6,1
4.9,2.4,3.3,1.0,1
6.6,2.9,4.6,1.3,1
5.2,2.7,3.9,1.4,1
5.0,2.0,3.5,1.0,1
5.9,3.0,4.2,1.5,1
6.0,2.2,4.0,1.0,1
6.1,2.9,4.7,1.4,1
5.6,2.9,3.6,1.3,1
6.7,3.1,4.4,1.4,1
5.6,3.0,4.5,1.5,1
5.8,2.7,4.1,1.0,1
6.2,2.2,4.5,1.5,1
5.6,2.5,3.9,1.1,1
5.9,3.2,4.8,1.8,1
6.1,2.8,4.0,1.3,1
6.3,2.5,4.9,1.5,1
6.1,2.8,4.7,1.2,1
6.4,2.9,4.3,1.3,1
6.6,3.0,4.4,1.4,1
6.8,2.8,4.8,1.4,1
6.7,3.0,5.0,1.7,1
6.0,2.9,4.5,1.5,1
5.7,2.6,3.5,1.0,1
5.5,2.4,3.8,1.1,1
5.5,2.4,3.7,1.0,1
5.8,2.7,3.9,1.2,1
6.0,2.7,5.1,1.6,1
5.4,3.0,4.5,1.5,1
6.0,3.4,4.5,1.6,1
6.7,3.1,4.7,1.5,1
6.3,2.3,4.4,1.3,1
5.6,3.0,4.1,1.3,1
5.5,2.5,4.0,1.3,1
5.5,2.6,4.4,1.2,1
6.1,3.0,4.6,1.4,1
5.8,2.6,4.0,1.2,1
5.0,2.3,3.3,1.0,1
5.6,2.7,4.2,1.3,1
5.7,3.0,4.2,1.2,1
5.7,2.9,4.2,1.3,1
6.2,2.9,4.3,1.3,1
5.1,2.5,3.0,1.1,1
5.7,2.8,4.1,1.3,1
6.3,3.3,6.0,2.5,2
5.8,2.7,5.1,1.9,2
7.1,3.0,5.9,2.1,2
6.3,2.9,5.6,1.8,2
6.5,3.0,5.8,2.2,2
7.6,3.0,6.6,2.1,2
4.9,2.5,4.5,1.7,2
7.3,2.9,6.3,1.8,2
6.7,2.5,5.8,1.8,2
7.2,3.6,6.1,2.5,2
6.5,3.2,5.1,2.0,2
6.4,2.7,5.3,1.9,2
6.8,3.0,5.5,2.1,2
5.7,2.5,5.0,2.0,2
5.8,2.8,5.1,2.4,2
6.4,3.2,5.3,2.3,2
6.5,3.0,5.5,1.8,2
7.7,3.8,6.7,2.2,2
7.7,2.6,6.9,2.3,2
6.0,2.2,5.0,1.5,2
6.9,3.2,5.7,2.3,2
5.6,2.8,4.9,2.0,2
7.7,2.8,6.7,2.0,2
6.3,2.7,4.9,1.8,2
6.7,3.3,5.7,2.1,2
7.2,3.2,6.0,1.8,2
6.2,2.8,4.8,1.8,2
6.1,3.0,4.9,1.8,2
6.4,2.8,5.6,2.1,2
7.2,3.0,5.8,1.6,2
7.4,2.8,6.1,1.9,2
7.9,3.8,6.4,2.0,2
6.4,2.8,5.6,2.2,2
6.3,2.8,5.1,1.5,2
6.1,2.6,5.6,1.4,2
7.7,3.0,6.1,2.3,2
6.3,3.4,5.6,2.4,2
6.4,3.1,5.5,1.8,2
6.0,3.0,4.8,1.8,2
6.9,3.1,5.4,2.1,2
6.7,3.1,5.6,2.4,2
6.9,3.1,5.1,2.3,2
5.8,2.7,5.1,1.9,2
6.8,3.2,5.9,2.3,2
6.7,3.3,5.7,2.5,2
6.7,3.0,5.2,2.3,2
6.3,2.5,5.0,1.9,2
6.5,3.0,5.2,2.0,2
6.2,3.4,5.4,2.3,2
5.9,3.0,5.1,1.8,2
CSV;
}
