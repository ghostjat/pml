<?php

declare(strict_types=1);

namespace Pml\Data;

use Pml\Dataset;
use Pml\Tensor;
use RuntimeException;

/**
 * Lazy, memory-bounded Dataset reader for large files.
 *
 * Reads CSV or NDJSON line-by-line and yields Dataset chunks of at most
 * $chunkSize rows.  Peak RAM usage is O(chunkSize × features × 4 bytes)
 * — the rest of the file stays on disk.
 *
 * Usage:
 *   $stream = new StreamingDataset('train.csv', chunkSize: 1000, labelColumn: 0);
 *   foreach ($stream->chunks() as $chunk) {
 *       $pipeline->train($chunk);
 *   }
 *
 * Supported formats: CSV (auto-detected by extension or forced via $format).
 * NDJSON support: each JSON object must contain a 'features' array key and
 * an optional 'label' scalar key.
 */
final class StreamingDataset
{
    private const FORMAT_CSV   = 'csv';
    private const FORMAT_NDJSON = 'ndjson';

    private string $format;

    /**
     * @param string   $path         Absolute or relative path to the data file.
     * @param int      $chunkSize    Max rows per yielded Dataset chunk.
     * @param int      $labelColumn  0-based CSV column used as label (-1 = none).
     *                               For NDJSON this is ignored; use the 'label' key.
     * @param bool     $hasHeader    Whether the CSV has a header row to skip.
     * @param string|null $format    'csv' | 'ndjson' | null (auto-detect by extension).
     */
    public function __construct(
        private readonly string $path,
        private readonly int $chunkSize = 512,
        private readonly int $labelColumn = -1,
        private readonly bool $hasHeader = true,
        ?string $format = null
    ) {
        if (!is_file($this->path)) {
            throw new RuntimeException("StreamingDataset: file not found '{$this->path}'.");
        }

        $this->format = $format ?? $this->detectFormat($path);
    }

    /**
     * Yields Dataset chunks until the file is exhausted.
     *
     * @return \Generator<int, Dataset>
     */
    public function chunks(): \Generator
    {
        return match ($this->format) {
            self::FORMAT_CSV    => $this->streamCsv(),
            self::FORMAT_NDJSON => $this->streamNdjson(),
            default             => throw new RuntimeException(
                "StreamingDataset: unsupported format '{$this->format}'."
            ),
        };
    }

    // -------------------------------------------------------------------------

    /**
     * Streams a CSV file chunk by chunk.
     * Memory: only $chunkSize rows are alive at any point.
     *
     * @return \Generator<int, Dataset>
     */
    private function streamCsv(): \Generator
    {
        $handle = fopen($this->path, 'r');
        if ($handle === false) {
            throw new RuntimeException("StreamingDataset: cannot open '{$this->path}'.");
        }

        try {
            if ($this->hasHeader) {
                fgetcsv($handle); // skip header
            }

            $chunkIndex  = 0;
            $rows        = [];
            $labelValues = [];

            while (($row = fgetcsv($handle)) !== false) {
                if ($this->labelColumn >= 0) {
                    $labelValues[] = (float) $row[$this->labelColumn];
                    unset($row[$this->labelColumn]);
                    $rows[] = array_values(array_map('floatval', $row));
                } else {
                    $rows[] = array_map('floatval', $row);
                }

                if (\count($rows) >= $this->chunkSize) {
                    yield $chunkIndex++ => $this->buildDataset($rows, $labelValues);
                    $rows        = [];
                    $labelValues = [];
                }
            }

            // Yield the last partial chunk (if any).
            if (!empty($rows)) {
                yield $chunkIndex => $this->buildDataset($rows, $labelValues);
            }
        } finally {
            fclose($handle);
        }
    }

    /**
     * Streams an NDJSON file chunk by chunk.
     * Each line must be a JSON object with a 'features' array and optional 'label'.
     *
     * @return \Generator<int, Dataset>
     */
    private function streamNdjson(): \Generator
    {
        $handle = fopen($this->path, 'r');
        if ($handle === false) {
            throw new RuntimeException("StreamingDataset: cannot open '{$this->path}'.");
        }

        try {
            $chunkIndex  = 0;
            $rows        = [];
            $labelValues = [];

            while (($line = fgets($handle)) !== false) {
                $line = trim($line);
                if ($line === '') continue;

                $obj = json_decode($line, true, 512, \JSON_THROW_ON_ERROR);

                if (!isset($obj['features']) || !\is_array($obj['features'])) {
                    throw new RuntimeException(
                        "StreamingDataset (NDJSON): each object must have a 'features' array."
                    );
                }

                $rows[] = array_map('floatval', $obj['features']);

                if (isset($obj['label'])) {
                    $labelValues[] = (float) $obj['label'];
                }

                if (\count($rows) >= $this->chunkSize) {
                    yield $chunkIndex++ => $this->buildDataset($rows, $labelValues);
                    $rows        = [];
                    $labelValues = [];
                }
            }

            if (!empty($rows)) {
                yield $chunkIndex => $this->buildDataset($rows, $labelValues);
            }
        } finally {
            fclose($handle);
        }
    }

    // -------------------------------------------------------------------------

    private function buildDataset(array $rows, array $labelValues): Dataset
    {
        $samples = Tensor::fromArray($rows);
        $labels  = !empty($labelValues) ? Tensor::fromArray($labelValues) : null;
        return new Dataset($samples, $labels);
    }

    private function detectFormat(string $path): string
    {
        $ext = strtolower(pathinfo($path, \PATHINFO_EXTENSION));
        return match ($ext) {
            'ndjson', 'jsonl' => self::FORMAT_NDJSON,
            default           => self::FORMAT_CSV,
        };
    }
}
