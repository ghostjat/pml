<?php

declare(strict_types=1);

namespace Pml\Classic\Datasets;

use Pml\Tensor;

/**
 * DataGenerator — Scikit-Learn-style synthetic dataset generators.
 *
 * All methods return [Tensor $X, Tensor $y] — destructure as:
 *   [$X, $y] = DataGenerator::make_blobs(n_samples: 300, centers: 3);
 *
 * Reproducibility:
 *   Every generator accepts a $seed parameter.  Internally we use a seeded
 *   LCG (Numerical Recipes constants) + Box-Muller transform, producing
 *   identical results across PHP versions and operating systems.
 *
 * sklearn equivalents:
 *   make_blobs()          → sklearn.datasets.make_blobs
 *   make_moons()          → sklearn.datasets.make_moons
 *   make_regression()     → sklearn.datasets.make_regression
 *   make_classification() → sklearn.datasets.make_classification
 */
final class DataGenerator
{
    // ── LCG constants (Numerical Recipes / glibc) ─────────────────────────────
    private const LCG_M = 4_294_967_296; // 2^32
    private const LCG_A = 1_664_525;
    private const LCG_C = 1_013_904_223;

    // ─────────────────────────────────────────────────────────────────────────
    //  make_blobs
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Generate isotropic Gaussian blobs for clustering / classification.
     *
     * Cluster k is centred at (k·5, k·5, …) in all $n_features dimensions.
     * With the default cluster_std=1.0 and sep=5 the clusters are ~5 σ apart —
     * easily separable by any distance-based algorithm.
     *
     * Samples are allocated in contiguous class blocks (labels 0, 1, 2, …).
     * If $n_samples is not divisible by $centers, the first few clusters get
     * one extra sample so the total is exactly $n_samples.
     *
     * @param int   $n_samples   Total number of samples.
     * @param int   $n_features  Dimensionality of each sample.
     * @param int   $centers     Number of cluster centres (classes).
     * @param float $cluster_std Standard deviation of each cluster (isotropic).
     * @param int   $seed        RNG seed for reproducibility.
     *
     * @return array{0: Tensor, 1: Tensor}  [$X[n_samples, n_features], $y[n_samples]]
     */
    public static function make_blobs(
        int   $n_samples   = 100,
        int   $n_features  = 2,
        int   $centers     = 3,
        float $cluster_std = 1.0,
        int   $seed        = 42,
    ): array {
        [$s1, $s2] = self::seeds($seed, 0, 1);

        $X   = new Tensor([$n_samples, $n_features]);
        $y   = new Tensor([$n_samples]);
        $sep = 5.0;

        $base      = intdiv($n_samples, $centers);
        $remainder = $n_samples % $centers;
        $idx       = 0;

        for ($k = 0; $k < $centers; $k++) {
            $count  = $base + ($k < $remainder ? 1 : 0);
            $center = (float) ($k * $sep);

            for ($i = 0; $i < $count; $i++) {
                for ($j = 0; $j < $n_features; $j++) {
                    $X->buffer[$idx * $n_features + $j] = $center + self::randn($s1, $s2) * $cluster_std;
                }
                $y->buffer[$idx] = (float) $k;
                $idx++;
            }
        }

        return [$X, $y];
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  make_moons
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Generate two interleaving half-circles (the classic "moons" dataset).
     *
     * Geometry:
     *   Moon 0:  x = cos(t),       y = sin(t),       t ∈ [0, π]  centred at (0, 0)
     *   Moon 1:  x = 1 − cos(t),   y = 0.5 − sin(t), t ∈ [0, π]  centred at (1, 0.5)
     *
     * Gaussian noise with std = $noise is added independently to each coordinate.
     * The dataset is NOT linearly separable, making it ideal for evaluating
     * kernel SVMs and non-linear classifiers.
     *
     * @param int   $n_samples Total number of points (split ≈ 50/50 between moons).
     * @param float $noise     Std-dev of Gaussian noise added to each coordinate.
     * @param int   $seed      RNG seed.
     *
     * @return array{0: Tensor, 1: Tensor}  [$X[n_samples, 2], $y[n_samples]]
     */
    public static function make_moons(
        int   $n_samples = 100,
        float $noise     = 0.1,
        int   $seed      = 42,
    ): array {
        [$s1, $s2] = self::seeds($seed, 3, 5);

        $n0  = intdiv($n_samples, 2);
        $n1  = $n_samples - $n0;

        $X   = new Tensor([$n_samples, 2]);
        $y   = new Tensor([$n_samples]);
        $idx = 0;

        // Moon 0 — upper half-circle, label = 0
        for ($i = 0; $i < $n0; $i++) {
            $t = M_PI * $i / max(1, $n0 - 1);
            $X->buffer[$idx * 2]     = (float) (cos($t) + self::randn($s1, $s2) * $noise);
            $X->buffer[$idx * 2 + 1] = (float) (sin($t) + self::randn($s1, $s2) * $noise);
            $y->buffer[$idx]         = 0.0;
            $idx++;
        }

        // Moon 1 — lower half-circle, shifted to (1, 0.5), label = 1
        for ($i = 0; $i < $n1; $i++) {
            $t = M_PI * $i / max(1, $n1 - 1);
            $X->buffer[$idx * 2]     = (float) (1.0 - cos($t) + self::randn($s1, $s2) * $noise);
            $X->buffer[$idx * 2 + 1] = (float) (0.5 - sin($t) + self::randn($s1, $s2) * $noise);
            $y->buffer[$idx]         = 1.0;
            $idx++;
        }

        return [$X, $y];
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  make_regression
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Generate a random regression problem.
     *
     * The first $n_informative features are drawn from N(0,1) and contribute
     * linearly to the target via a random weight vector w ~ N(0, 5²).
     * The remaining $n_features − $n_informative features are pure N(0,1)
     * noise with no contribution to y.
     *
     *   y = X[:, :n_informative] · w + ε,   ε ~ N(0, noise²)
     *
     * The high signal-to-noise ratio (|w| ~ 5, σ_noise ≪ |w|) makes this
     * dataset well-suited for validating regularised linear models.
     *
     * @param int   $n_samples     Number of samples.
     * @param int   $n_features    Total number of features (including noise features).
     * @param int   $n_informative Number of features that actually affect y.
     * @param float $noise         Std-dev of Gaussian noise added to y.
     * @param int   $seed          RNG seed.
     *
     * @return array{0: Tensor, 1: Tensor}  [$X[n_samples, n_features], $y[n_samples]]
     */
    public static function make_regression(
        int   $n_samples     = 100,
        int   $n_features    = 10,
        int   $n_informative = 5,
        float $noise         = 0.0,
        int   $seed          = 42,
    ): array {
        $n_informative = min($n_informative, $n_features);
        [$s1, $s2]     = self::seeds($seed, 7, 11);

        // Random weight vector for the informative features
        $weights = [];
        for ($j = 0; $j < $n_informative; $j++) {
            $weights[$j] = self::randn($s1, $s2) * 5.0;
        }

        $X = new Tensor([$n_samples, $n_features]);
        $y = new Tensor([$n_samples]);

        for ($i = 0; $i < $n_samples; $i++) {
            $yi = 0.0;

            // Informative features — each contributes to y
            for ($j = 0; $j < $n_informative; $j++) {
                $xij = self::randn($s1, $s2);
                $X->buffer[$i * $n_features + $j] = $xij;
                $yi += $xij * $weights[$j];
            }

            // Pure noise features — no signal
            for ($j = $n_informative; $j < $n_features; $j++) {
                $X->buffer[$i * $n_features + $j] = self::randn($s1, $s2);
            }

            // Target noise ε
            if ($noise > 0.0) {
                $yi += self::randn($s1, $s2) * $noise;
            }

            $y->buffer[$i] = $yi;
        }

        return [$X, $y];
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  make_classification
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Generate a balanced n-class classification problem from Gaussian blobs.
     *
     * Class k is centred at (k·$class_sep, k·$class_sep, …) in all feature
     * dimensions.  With class_sep=5.0 and cluster_std=0.8 the nearest-class
     * margin is >6 σ — easily linearly separable for any standard classifier.
     *
     * Samples are balanced: each class gets ⌊n_samples/n_classes⌋ rows, with
     * the first (n_samples mod n_classes) classes getting one extra sample.
     *
     * @param int   $n_samples   Total number of samples.
     * @param int   $n_features  Dimensionality.
     * @param int   $n_classes   Number of classes.
     * @param float $class_sep   Distance between consecutive class centroids (per dimension).
     * @param float $cluster_std Within-class standard deviation (isotropic).
     * @param int   $seed        RNG seed.
     *
     * @return array{0: Tensor, 1: Tensor}  [$X[n_samples, n_features], $y[n_samples]]
     */
    public static function make_classification(
        int   $n_samples   = 100,
        int   $n_features  = 20,
        int   $n_classes   = 2,
        float $class_sep   = 5.0,
        float $cluster_std = 0.8,
        int   $seed        = 42,
    ): array {
        [$s1, $s2] = self::seeds($seed, 17, 31);

        $X = new Tensor([$n_samples, $n_features]);
        $y = new Tensor([$n_samples]);

        $base      = intdiv($n_samples, $n_classes);
        $remainder = $n_samples % $n_classes;
        $idx       = 0;

        for ($k = 0; $k < $n_classes; $k++) {
            $count  = $base + ($k < $remainder ? 1 : 0);
            $center = (float) ($k * $class_sep);

            for ($i = 0; $i < $count; $i++) {
                for ($j = 0; $j < $n_features; $j++) {
                    $X->buffer[$idx * $n_features + $j] = $center + self::randn($s1, $s2) * $cluster_std;
                }
                $y->buffer[$idx] = (float) $k;
                $idx++;
            }
        }

        return [$X, $y];
    }

    // ── Private RNG helpers ───────────────────────────────────────────────────

    /**
     * Initialise two independent LCG states from a seed + two small offsets.
     *
     * Using different offsets for each generator function prevents them from
     * producing the same initial sequence when called with the same seed.
     *
     * @return array{0: int, 1: int}
     */
    private static function seeds(int $seed, int $o1, int $o2): array
    {
        return [
            ($seed + $o1) & 0xFFFFFFFF,
            ($seed + $o2) & 0xFFFFFFFF,
        ];
    }

    /**
     * Advance one LCG stream by a single step and return a uniform float in (0, 1].
     */
    private static function lcg(int &$s): float
    {
        $s = ((self::LCG_A * $s + self::LCG_C) & 0xFFFFFFFF);
        return $s / self::LCG_M;
    }

    /**
     * Box-Muller transform: return one standard-normal variate N(0,1) using
     * two independent LCG streams.
     *
     * Note: We use only the cosine branch (discard the sine) to avoid the need
     * for static state that would break determinism across call boundaries.
     * The sin branch's entropy is implicitly preserved in the LCG state.
     */
    private static function randn(int &$s1, int &$s2): float
    {
        $u1 = max(self::lcg($s1), 1e-10); // guard against log(0)
        $u2 = self::lcg($s2);
        return (float) (sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2));
    }
}
