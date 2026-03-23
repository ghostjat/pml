<?php

declare(strict_types=1);

namespace Pml\Classic\Datasets;

use Pml\Tensor;

/**
 * DataLoader — Scikit-Learn-style CSV ingestion utility.
 *
 * Mirrors sklearn's load_*() convention by returning a "Bunch"-style
 * associative array with keys: 'data', 'target', 'feature_names'.
 *
 * Usage:
 *   $bunch = DataLoader::load_csv('path/to/iris.csv', target_column: 'species');
 *
 *   $X             = $bunch['data'];           // Tensor[n,d] or list<list<float|string>>
 *   $y             = $bunch['target'];          // Tensor[n] or list<float|string> or null
 *   $featureNames  = $bunch['feature_names'];   // list<string>
 *
 * Column types:
 *   • All-numeric columns → values cast to float.
 *   • Mixed/string columns → values kept as string (pass to OneHotEncoder later).
 *
 * If every X column is numeric, 'data' is a Tensor; otherwise a 2D PHP array.
 * If the target column is all-numeric, 'target' is a Tensor; otherwise a PHP array.
 */
final class DataLoader
{
    /**
     * Load a CSV file and split into features (X) and target (y).
     *
     * @param string          $filepath      Absolute or relative path to the CSV.
     * @param string|int|null $target_column Column name (when $header = true) or 0-based
     *                                       column index. Null → no target column extracted.
     * @param bool            $header        True if the first row is a header row.
     *
     * @return array{
     *   data:          Tensor|list<list<float|string>>,
     *   target:        Tensor|list<float|string>|null,
     *   feature_names: list<string>
     * }
     *
     * @throws \RuntimeException        If the file cannot be opened or contains no data rows.
     * @throws \InvalidArgumentException If $target_column (string) is not present in the header.
     */
    public static function load_csv(
        string          $filepath,
        string|int|null $target_column = null,
        bool            $header        = true,
    ): array {
        if (!is_file($filepath) || !is_readable($filepath)) {
            throw new \RuntimeException("CSV file not found or not readable: {$filepath}");
        }

        $fh = fopen($filepath, 'r');
        if ($fh === false) {
            throw new \RuntimeException("Failed to open: {$filepath}");
        }

        try {
            return self::parse($fh, $target_column, $header);
        } finally {
            fclose($fh);
        }
    }

    /**
     * Load a JSON file (array of objects) and split into features and target.
     *
     * The file must contain a top-level JSON array of objects, where each
     * object is one row and the object's keys are column names.  This is
     * the format produced by most REST APIs and `pandas.DataFrame.to_json(orient='records')`.
     *
     * @param string      $filepath      Absolute or relative path to the .json file.
     * @param string|null $target_column Column name to extract as the target vector.
     *                                   Null → no target column extracted.
     *
     * @return array{
     *   data:          Tensor|list<list<float|string>>,
     *   target:        Tensor|list<float|string>|null,
     *   feature_names: list<string>
     * }
     *
     * @throws \RuntimeException        If the file cannot be opened or is not a JSON array.
     * @throws \InvalidArgumentException If $target_column is not present in the data.
     */
    public static function load_json(
        string  $filepath,
        ?string $target_column = null,
    ): array {
        if (!is_file($filepath) || !is_readable($filepath)) {
            throw new \RuntimeException("JSON file not found or not readable: {$filepath}");
        }

        $raw = file_get_contents($filepath);
        if ($raw === false) {
            throw new \RuntimeException("Failed to read: {$filepath}");
        }

        $decoded = json_decode($raw, true);
        if (!is_array($decoded) || (count($decoded) > 0 && !is_array($decoded[0]))) {
            throw new \RuntimeException(
                "load_json: expected a JSON array of objects in '{$filepath}'."
            );
        }

        return self::parseObjects($decoded, $target_column);
    }

    /**
     * Load an NDJSON (Newline-Delimited JSON) file using a streaming generator.
     *
     * Each line in the file must be a single valid JSON object.  Lines are read
     * and decoded one at a time, so only one parsed object is in PHP memory at
     * a time — ideal for datasets that would exhaust RAM if loaded via load_json().
     *
     * Blank lines and lines that fail json_decode() are silently skipped.
     *
     * @param string      $filepath      Absolute or relative path to the .ndjson file.
     * @param string|null $target_column Column name to extract as the target vector.
     *
     * @return array{
     *   data:          Tensor|list<list<float|string>>,
     *   target:        Tensor|list<float|string>|null,
     *   feature_names: list<string>
     * }
     *
     * @throws \RuntimeException        If the file cannot be opened or contains no valid rows.
     * @throws \InvalidArgumentException If $target_column is not present in the data.
     */
    public static function load_ndjson(
        string  $filepath,
        ?string $target_column = null,
    ): array {
        if (!is_file($filepath) || !is_readable($filepath)) {
            throw new \RuntimeException("NDJSON file not found or not readable: {$filepath}");
        }

        $fh = fopen($filepath, 'r');
        if ($fh === false) {
            throw new \RuntimeException("Failed to open: {$filepath}");
        }

        try {
            return self::parseObjects(self::ndjsonGenerator($fh), $target_column);
        } finally {
            fclose($fh);
        }
    }

    // ── Internal parser ───────────────────────────────────────────────────────

    /** @param resource $fh */
    private static function parse(
        mixed           $fh,
        string|int|null $target_column,
        bool            $header,
    ): array {
        // ── Header row ─────────────────────────────────────────────────────────
        $colNames = [];
        if ($header) {
            $row = fgetcsv($fh);
            if ($row === false || $row === null) {
                throw new \RuntimeException('CSV is empty (no header row found).');
            }
            $colNames = array_map('trim', $row);
        }

        // ── Resolve target column index ────────────────────────────────────────
        $targetIdx = null;
        if ($target_column !== null) {
            if (is_int($target_column)) {
                $targetIdx = $target_column;
            } else {
                $idx = array_search($target_column, $colNames, true);
                if ($idx === false) {
                    throw new \InvalidArgumentException(
                        "Target column '{$target_column}' not found. "
                        . 'Available: ' . implode(', ', $colNames)
                    );
                }
                $targetIdx = (int) $idx;
            }
        }

        // ── Stream rows into raw PHP arrays ────────────────────────────────────
        $Xrows = [];
        $yvals = [];
        $nCols = null;

        while (($row = fgetcsv($fh)) !== false) {
            if ($row === [null]) {
                continue; // blank line
            }
            $row = array_map('trim', $row);

            // Auto-build column names if no header was provided
            if ($nCols === null) {
                $nCols = count($row);
                if (!$header) {
                    for ($j = 0; $j < $nCols; $j++) {
                        $colNames[] = "x{$j}";
                    }
                }
            }

            // Cast each cell to float where possible; keep strings otherwise
            $typed = [];
            foreach ($row as $cell) {
                $typed[] = is_numeric($cell) && $cell !== '' ? (float) $cell : $cell;
            }

            if ($targetIdx !== null) {
                $yvals[] = $typed[$targetIdx];
                $xrow    = [];
                foreach ($typed as $j => $v) {
                    if ($j !== $targetIdx) {
                        $xrow[] = $v;
                    }
                }
                $Xrows[] = $xrow;
            } else {
                $Xrows[] = $typed;
            }
        }

        if (count($Xrows) === 0) {
            throw new \RuntimeException('CSV contains no data rows.');
        }

        // ── Feature names (excluding target column) ────────────────────────────
        $featureNames = [];
        foreach ($colNames as $j => $name) {
            if ($j !== $targetIdx) {
                $featureNames[] = $name;
            }
        }

        // ── Build X ────────────────────────────────────────────────────────────
        //
        // If every value in every row is a float, pack into a Tensor for
        // zero-copy compatibility with Pml estimators.
        // Otherwise return as a 2D PHP array so the caller can apply
        // string encoding (e.g. OneHotEncoder) before constructing a Tensor.
        $xAllNumeric = true;
        foreach ($Xrows as $xrow) {
            foreach ($xrow as $v) {
                if (!is_float($v)) {
                    $xAllNumeric = false;
                    break 2;
                }
            }
        }

        if ($xAllNumeric) {
            $n    = count($Xrows);
            $d    = count($Xrows[0]);
            $flat = [];
            foreach ($Xrows as $xrow) {
                foreach ($xrow as $v) {
                    $flat[] = (float) $v;
                }
            }
            $X = Tensor::fromArray($flat, [$n, $d]);
        } else {
            $X = $Xrows; // 2D PHP array — caller must encode strings
        }

        // ── Build y ────────────────────────────────────────────────────────────
        if ($targetIdx === null || count($yvals) === 0) {
            $y = null;
        } else {
            $yAllNumeric = true;
            foreach ($yvals as $v) {
                if (!is_float($v)) {
                    $yAllNumeric = false;
                    break;
                }
            }
            $y = $yAllNumeric
                ? Tensor::fromArray(array_map('floatval', $yvals), [count($yvals)])
                : $yvals;
        }

        return [
            'data'          => $X,
            'target'        => $y,
            'feature_names' => $featureNames,
        ];
    }

    // ── JSON helpers ──────────────────────────────────────────────────────────

    /**
     * Stream NDJSON lines from a file handle as decoded PHP arrays.
     *
     * Each call to fgets() reads one line; json_decode() parses it immediately.
     * This ensures that only one raw line and one decoded object are in PHP
     * memory at a time — the key property that makes NDJSON memory-efficient.
     * Blank lines and malformed JSON lines are silently skipped.
     *
     * @param  resource $fh  Open file handle positioned at the start of the data.
     * @return \Generator<int, array<string, mixed>>
     */
    private static function ndjsonGenerator(mixed $fh): \Generator
    {
        while (($line = fgets($fh)) !== false) {
            $line = trim($line);
            if ($line === '') {
                continue;
            }
            $obj = json_decode($line, true);
            if (is_array($obj)) {
                yield $obj;
            }
            // Malformed lines are skipped silently
        }
    }

    /**
     * Parse an iterable of row objects (assoc arrays) into a Bunch-style array.
     *
     * This is the shared post-decode processing step used by both load_json()
     * and load_ndjson().  It mirrors the type-detection and Tensor-building
     * logic of parse() but reads from an iterable instead of a file handle.
     *
     * @param  iterable<array<string, mixed>> $rows         Stream of decoded row objects.
     * @param  string|null                    $target_column Column name to extract as y.
     *
     * @return array{
     *   data:          Tensor|list<list<float|string>>,
     *   target:        Tensor|list<float|string>|null,
     *   feature_names: list<string>
     * }
     */
    private static function parseObjects(iterable $rows, ?string $target_column): array
    {
        $colNames  = null;
        $targetIdx = null;
        $Xrows     = [];
        $yvals     = [];

        foreach ($rows as $rowObj) {
            // ── Discover column names from the first row ────────────────────
            if ($colNames === null) {
                $colNames = array_keys($rowObj);

                if ($target_column !== null) {
                    $idx = array_search($target_column, $colNames, true);
                    if ($idx === false) {
                        throw new \InvalidArgumentException(
                            "Target column '{$target_column}' not found. "
                            . 'Available: ' . implode(', ', $colNames)
                        );
                    }
                    $targetIdx = (int) $idx;
                }
            }

            // ── Cast each cell (numeric string → float, else keep as string) ─
            $typed = [];
            foreach ($rowObj as $v) {
                $s       = (string) $v;
                $typed[] = (is_numeric($s) && $s !== '') ? (float) $s : $s;
            }

            if ($targetIdx !== null) {
                $yvals[] = $typed[$targetIdx];
                $xrow    = [];
                foreach ($typed as $j => $v) {
                    if ($j !== $targetIdx) {
                        $xrow[] = $v;
                    }
                }
                $Xrows[] = $xrow;
            } else {
                $Xrows[] = $typed;
            }
        }

        if (count($Xrows) === 0) {
            throw new \RuntimeException('Data source contains no rows.');
        }

        // ── Feature names (excluding target column) ─────────────────────────
        $featureNames = [];
        if ($colNames !== null) {
            foreach ($colNames as $j => $name) {
                if ($j !== $targetIdx) {
                    $featureNames[] = $name;
                }
            }
        }

        // ── Build X ─────────────────────────────────────────────────────────
        $xAllNumeric = true;
        foreach ($Xrows as $xrow) {
            foreach ($xrow as $v) {
                if (!is_float($v)) {
                    $xAllNumeric = false;
                    break 2;
                }
            }
        }

        if ($xAllNumeric) {
            $n    = count($Xrows);
            $d    = count($Xrows[0]);
            $flat = [];
            foreach ($Xrows as $xrow) {
                foreach ($xrow as $v) {
                    $flat[] = (float) $v;
                }
            }
            $X = Tensor::fromArray($flat, [$n, $d]);
        } else {
            $X = $Xrows;
        }

        // ── Build y ─────────────────────────────────────────────────────────
        if ($targetIdx === null || count($yvals) === 0) {
            $y = null;
        } else {
            $yAllNumeric = true;
            foreach ($yvals as $v) {
                if (!is_float($v)) {
                    $yAllNumeric = false;
                    break;
                }
            }
            $y = $yAllNumeric
                ? Tensor::fromArray(array_map('floatval', $yvals), [count($yvals)])
                : $yvals;
        }

        return [
            'data'          => $X,
            'target'        => $y,
            'feature_names' => $featureNames,
        ];
    }
}
