<?php

declare(strict_types=1);

namespace Pml\Lib;

use Pml\Tensor;

/**
 * HuggingFace-compatible SafeTensors I/O bridge.
 *
 * File layout (SafeTensors v1 spec):
 *   [8 bytes] uint64-LE  — byte length of JSON header
 *   [N bytes] UTF-8 JSON — tensor metadata, padded to 8-byte alignment with spaces
 *   [D bytes] raw data   — tensors written contiguously in JSON declaration order
 *
 * PHP never touches tensor bytes:
 *   - save() passes the header + C-pointers to the kernel; data is written by C.
 *   - load() maps tensor regions via mmap; PHP receives a Tensor backed by the OS
 *     page cache with zero bytes copied into PHP or C heap.
 */
final class SafeTensorsIO
{
    /** Maps Pml dtype int → SafeTensors dtype string */
    private const DTYPE_TO_ST = [
        Tensor::DTYPE_FLOAT32 => 'F32',
        Tensor::DTYPE_INT32   => 'I32',
        Tensor::DTYPE_INT64   => 'I64',
    ];

    /** Maps SafeTensors dtype string → Pml dtype int (superset for HF interop) */
    private const ST_TO_DTYPE = [
        'F32'  => Tensor::DTYPE_FLOAT32,
        'F16'  => Tensor::DTYPE_FLOAT32,   // up-cast on load; no F16 engine support
        'BF16' => Tensor::DTYPE_FLOAT32,   // same
        'I32'  => Tensor::DTYPE_INT32,
        'I64'  => Tensor::DTYPE_INT64,
    ];

    /** Bytes-per-element for each Pml dtype */
    private const ELEM_BYTES = [
        Tensor::DTYPE_FLOAT32 => 4,
        Tensor::DTYPE_INT32   => 4,
        Tensor::DTYPE_INT64   => 8,
    ];

    // -------------------------------------------------------------------------

    /**
     * Serialize named tensors to a HF-compatible SafeTensors file.
     *
     * The JSON header is built entirely in PHP (pure arithmetic on tensor metadata),
     * then handed to the C kernel together with the raw TensorC* array.
     * The kernel writes [header-length][header][data] without involving PHP memory.
     *
     * @param string             $filepath Output path.
     * @param array<string,Tensor> $tensors  Name → Tensor map (insertion order preserved).
     */
    public static function save(string $filepath, array $tensors): void
    {
        if (empty($tensors)) {
            throw new \InvalidArgumentException('SafeTensorsIO::save: tensor map is empty.');
        }

        $offset  = 0;
        $meta    = ['__metadata__' => ['format' => 'pt']];
        $ordered = [];

        foreach ($tensors as $name => $tensor) {
            if (!$tensor instanceof Tensor) {
                throw new \InvalidArgumentException("SafeTensorsIO::save: value for '$name' is not a Tensor.");
            }

            $dtype   = $tensor->ptr->dtype;
            $stDtype = self::DTYPE_TO_ST[$dtype] ?? 'F32';
            $ndim    = $tensor->ptr->ndim;

            $shape = [];
            for ($i = 0; $i < $ndim; $i++) {
                $shape[] = $tensor->ptr->shape[$i];
            }

            // byte_size is set by the C engine: total_size * sizeof(element)
            $byteLen = (int) $tensor->ptr->byte_size;

            $meta[$name] = [
                'dtype'        => $stDtype,
                'shape'        => $shape,
                'data_offsets' => [$offset, $offset + $byteLen],
            ];

            $offset  += $byteLen;
            $ordered[] = $tensor;
        }

        // Build JSON and pad to 8-byte boundary with trailing spaces (spec requirement)
        $header = json_encode($meta, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        $rem    = strlen($header) % 8;
        if ($rem !== 0) {
            $header .= str_repeat(' ', 8 - $rem);
        }

        $rc = Tensor::saveSafetensors($filepath, $header, $ordered);
        if ($rc === 0) {
            throw new \RuntimeException("SafeTensorsIO::save failed writing '$filepath'.");
        }
    }

    /**
     * Load a SafeTensors file as zero-copy, mmap-backed Tensor objects.
     *
     * Every returned Tensor is backed directly by the file's page-cache mapping;
     * no bytes are allocated or copied into C heap or PHP memory.
     * The caller is responsible for calling mmapFree() on each tensor when done,
     * or they will be reclaimed at process exit.
     *
     * @param  string $filepath
     * @return array<string, Tensor>  Name → mmap-backed Tensor.
     */
    public static function load(string $filepath): array
    {
        if (!is_file($filepath)) {
            throw new \RuntimeException("SafeTensorsIO::load: file not found '$filepath'.");
        }

        $fh = fopen($filepath, 'rb');
        if ($fh === false) {
            throw new \RuntimeException("SafeTensorsIO::load: cannot open '$filepath'.");
        }

        // --- Read 8-byte little-endian uint64 header length ---
        $raw = fread($fh, 8);
        if (strlen($raw) < 8) {
            fclose($fh);
            throw new \RuntimeException("SafeTensorsIO::load: file '$filepath' is truncated.");
        }

        // Unpack as two 32-bit LE words to guarantee little-endian on any platform
        ['lo' => $lo, 'hi' => $hi] = unpack('Vlo/Vhi', $raw);
        $headerLen = ($lo | ($hi << 32));

        // --- Read JSON header ---
        $json = fread($fh, $headerLen);
        fclose($fh);

        if (strlen($json) < $headerLen) {
            throw new \RuntimeException("SafeTensorsIO::load: header truncated in '$filepath'.");
        }

        /** @var array<string, mixed> $header */
        $header = json_decode($json, true, 512, JSON_THROW_ON_ERROR);

        // Data section begins immediately after the 8-byte prefix + padded header
        $dataBase = 8 + $headerLen;

        $result = [];

        foreach ($header as $name => $entry) {
            if ($name === '__metadata__') {
                continue;
            }
            if (!is_array($entry) ||
                !isset($entry['dtype'], $entry['shape'], $entry['data_offsets'])) {
                continue;
            }

            $dtype     = self::ST_TO_DTYPE[$entry['dtype']] ?? Tensor::DTYPE_FLOAT32;
            $shape     = array_map('intval', $entry['shape']);
            $byteStart = (int) $entry['data_offsets'][0];
            $offset    = $dataBase + $byteStart;

            // Zero-copy: maps file region directly; no PHP/C allocation
            $result[$name] = Tensor::fromMmap($filepath, $offset, $shape, $dtype);
        }

        return $result;
    }
}
