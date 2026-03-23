<?php

declare(strict_types=1);

namespace Pml\Classic\Exploration;

// ═══════════════════════════════════════════════════════════════════════════
//  DataProfiler — Exploratory Data Analysis (EDA) utility
//
//  Provides DataFrame-style inspection of a raw 2D PHP array, mirroring
//  the two most commonly used Pandas inspection functions:
//
//    info()     → pandas.DataFrame.info()
//    describe() → pandas.DataFrame.describe()
//
//  Both methods accept the 2D PHP array format returned by
//  DataLoader::load_csv() when the dataset contains string columns
//  (i.e. $bunch['data'] is an array, not a Tensor).
//
//  ── Separation of concerns ────────────────────────────────────────────────
//
//  DataProfiler is intentionally separate from DataLoader.  Its sole purpose
//  is to give the data scientist a human-readable snapshot of the raw data
//  BEFORE any transformation.  It reads but never modifies $X.
//
//  ── Usage ────────────────────────────────────────────────────────────────
//
//    $bunch = DataLoader::load_csv('titanic.csv', target_column: 'Survived');
//    DataProfiler::info($bunch['data'], $bunch['feature_names']);
//    $stats = DataProfiler::describe($bunch['data'], $bunch['feature_names']);
//    DataProfiler::print_describe($stats);
// ═══════════════════════════════════════════════════════════════════════════

final class DataProfiler
{
    // ── info() ────────────────────────────────────────────────────────────

    /**
     * Print a concise summary of a 2D PHP array, mirroring `df.info()`.
     *
     * For each column, prints:
     *   - Column index
     *   - Column name (from $feature_names, or "x{j}" if not provided)
     *   - Non-null count (cells that are not null and not empty string)
     *   - Inferred dtype: 'float64' if all non-null values are numeric,
     *     'object' otherwise (i.e. the column contains strings)
     *
     * @param array  $X             Raw 2D PHP array [n_samples][n_cols].
     * @param array  $feature_names Column names parallel to each row's indices.
     *                              Defaults to x0, x1, … if not provided.
     */
    public static function info(array $X, array $feature_names = []): void
    {
        $n = count($X);
        $d = ($n > 0) ? count($X[0]) : 0;

        // ── Dataset header ─────────────────────────────────────────────
        printf("<class 'Pml\\Classic\\Datasets\\DataLoader'>\n");
        printf("RangeIndex: %d entries, 0 to %d\n", $n, max(0, $n - 1));
        printf("Data columns (total %d columns):\n", $d);

        // ── Column table header ────────────────────────────────────────
        $w = [5, 22, 17, 10]; // column widths: #, Column, Non-Null Count, Dtype
        printf("%-{$w[0]}s %-{$w[1]}s %-{$w[2]}s %s\n", '#', 'Column', 'Non-Null Count', 'Dtype');
        printf("%-{$w[0]}s %-{$w[1]}s %-{$w[2]}s %s\n", '---', '------', '--------------', '-----');

        // ── Per-column rows ────────────────────────────────────────────
        for ($j = 0; $j < $d; $j++) {
            $name      = $feature_names[$j] ?? "x{$j}";
            $nonNull   = 0;
            $isNumeric = true;

            foreach ($X as $row) {
                $v = $row[$j] ?? null;
                if ($v !== null && $v !== '') {
                    $nonNull++;
                    if (!is_numeric($v)) {
                        $isNumeric = false;
                    }
                }
            }

            $dtype  = $isNumeric ? 'float64' : 'object';
            $nullStr = "{$nonNull} non-null";

            printf(
                "%-{$w[0]}d %-{$w[1]}s %-{$w[2]}s %s\n",
                $j,
                substr($name, 0, $w[1] - 1),
                $nullStr,
                $dtype
            );
        }

        printf("dtypes: float64(%d), object(%d)\n",
            self::countByDtype($X, $d, $feature_names, 'float64'),
            self::countByDtype($X, $d, $feature_names, 'object')
        );
    }

    // ── describe() ────────────────────────────────────────────────────────

    /**
     * Compute descriptive statistics for numeric columns, mirroring `df.describe()`.
     *
     * Only columns where all non-null values are numeric are included.
     * String/object columns are silently skipped (identical to Pandas default).
     *
     * Returns an associative array indexed by column name:
     *   $stats['Age'] = [
     *     'count' => 714.0,
     *     'mean'  => 29.699,
     *     'std'   => 14.526,
     *     'min'   => 0.42,
     *     '25%'   => 20.125,
     *     '50%'   => 28.0,
     *     '75%'   => 38.0,
     *     'max'   => 80.0,
     *   ]
     *
     * @param array  $X             Raw 2D PHP array [n_samples][n_cols].
     * @param array  $feature_names Column names parallel to each row's indices.
     *
     * @return array<string, array{count: float, mean: float, std: float,
     *                            min: float, '25%': float, '50%': float,
     *                            '75%': float, max: float}>
     */
    public static function describe(array $X, array $feature_names = []): array
    {
        $n    = count($X);
        $d    = ($n > 0) ? count($X[0]) : 0;
        $stats = [];

        for ($j = 0; $j < $d; $j++) {
            $name = $feature_names[$j] ?? "x{$j}";

            // ── Collect numeric values for this column ─────────────────
            $values = [];
            foreach ($X as $row) {
                $v = $row[$j] ?? null;
                if ($v !== null && $v !== '' && is_numeric($v)) {
                    $values[] = (float) $v;
                }
            }

            if (empty($values)) {
                continue; // non-numeric column — skip (Pandas behaviour)
            }

            $cnt = count($values);
            $sum = array_sum($values);
            $mean = $sum / $cnt;

            // ── Sample standard deviation (ddof=1, matching Pandas/NumPy) ──
            $variance = 0.0;
            foreach ($values as $v) {
                $variance += ($v - $mean) ** 2;
            }
            $std = ($cnt > 1) ? sqrt($variance / ($cnt - 1)) : 0.0;

            sort($values); // ascending for percentile computation

            $stats[$name] = [
                'count' => (float) $cnt,
                'mean'  => $mean,
                'std'   => $std,
                'min'   => $values[0],
                '25%'   => self::percentile($values, 0.25),
                '50%'   => self::percentile($values, 0.50),
                '75%'   => self::percentile($values, 0.75),
                'max'   => $values[$cnt - 1],
            ];
        }

        return $stats;
    }

    // ── print_describe() ──────────────────────────────────────────────────

    /**
     * Print the result of describe() as an ASCII table.
     *
     * Column widths are dynamically sized to the longer of the column name
     * and the widest formatted number.  Statistics appear as rows (matching
     * the Pandas vertical layout).
     *
     * @param array $stats        Output of describe().
     * @param int   $precision    Decimal places for numeric values (default 2).
     */
    public static function print_describe(array $stats, int $precision = 2): void
    {
        if (empty($stats)) {
            echo "(no numeric columns to describe)\n";
            return;
        }

        $statKeys = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
        $cols     = array_keys($stats);

        // ── Compute column widths ──────────────────────────────────────
        $labelW = max(6, max(array_map('strlen', $statKeys)));
        $colW   = [];
        foreach ($cols as $col) {
            $w = strlen($col);
            foreach ($statKeys as $sk) {
                $v = $stats[$col][$sk] ?? NAN;
                $w = max($w, strlen(sprintf("%.{$precision}f", $v)));
            }
            $colW[$col] = max($w + 2, 10);
        }

        // ── Header row ─────────────────────────────────────────────────
        printf("%-{$labelW}s", '');
        foreach ($cols as $col) {
            printf("%{$colW[$col]}s", $col);
        }
        echo "\n";

        // ── Separator ──────────────────────────────────────────────────
        $totalW = $labelW + array_sum($colW);
        echo str_repeat('-', $totalW) . "\n";

        // ── Statistic rows ─────────────────────────────────────────────
        foreach ($statKeys as $sk) {
            printf("%-{$labelW}s", $sk);
            foreach ($cols as $col) {
                $v = $stats[$col][$sk] ?? NAN;
                printf("%{$colW[$col]}.{$precision}f", $v);
            }
            echo "\n";
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * Linear-interpolation percentile on a sorted float array.
     *
     * Matches NumPy's default `linear` interpolation method — identical to
     * Pandas df.describe() output.
     *
     * @param  float[] $sorted  Sorted ascending float array (non-empty).
     * @param  float   $q       Quantile in [0, 1].
     * @return float
     */
    private static function percentile(array $sorted, float $q): float
    {
        $n = count($sorted);
        if ($n === 1) {
            return $sorted[0];
        }
        $idx  = $q * ($n - 1);
        $lo   = (int) floor($idx);
        $hi   = (int) ceil($idx);
        if ($lo === $hi) {
            return $sorted[$lo];
        }
        $frac = $idx - $lo;
        return $sorted[$lo] * (1.0 - $frac) + $sorted[$hi] * $frac;
    }

    /**
     * Count columns of a given inferred dtype ('float64' or 'object').
     */
    private static function countByDtype(array $X, int $d, array $featureNames, string $dtype): int
    {
        $n   = count($X);
        $cnt = 0;
        for ($j = 0; $j < $d; $j++) {
            $isNumeric = true;
            foreach ($X as $row) {
                $v = $row[$j] ?? null;
                if ($v !== null && $v !== '' && !is_numeric($v)) {
                    $isNumeric = false;
                    break;
                }
            }
            $colDtype = $isNumeric ? 'float64' : 'object';
            if ($colDtype === $dtype) {
                $cnt++;
            }
        }
        return $cnt;
    }
}
