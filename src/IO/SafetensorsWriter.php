<?php

declare(strict_types=1);

namespace Pml\IO;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  SafetensorsWriter
//
//  Serialises a named collection of Float32 tensors to the HuggingFace
//  .safetensors format, streaming each tensor's binary data one at a time
//  directly from its FFI C-memory buffer — no large PHP-string intermediary.
//
//  File format (spec: https://huggingface.co/docs/safetensors):
//
//   ┌──────────────────────────────────┐
//   │  8 bytes  │ uint64 LE header_len │
//   ├──────────────────────────────────┤
//   │ header_len bytes │ UTF-8 JSON    │
//   ├──────────────────────────────────┤
//   │ binary data (tensor buffers)     │
//   └──────────────────────────────────┘
//
//  JSON header structure per tensor:
//    {
//      "tensor_name": {
//        "dtype":        "F32",
//        "shape":        [dim0, dim1, ...],
//        "data_offsets": [byte_start, byte_end]   // relative to binary region
//      },
//      "__metadata__": { "key": "value", ... }    // optional
//    }
//
//  Byte-offset arithmetic:
//    Each float32 element is exactly 4 bytes (IEEE 754 single-precision).
//    Tensors are packed sequentially with no alignment padding:
//
//      tensor[0]  → bytes [0,           byteLen[0])
//      tensor[1]  → bytes [byteLen[0],  byteLen[0]+byteLen[1])
//      tensor[k]  → bytes [Σ byteLen[0..k-1],  Σ byteLen[0..k])
//
//    data_offsets[1] (exclusive end) equals the next tensor's data_offsets[0].
//
//  Memory strategy:
//    SafetensorsLoader::save() (the existing utility) collects all binary
//    chunks in a PHP array before writing, peaking at O(total_model_size).
//    SafetensorsWriter::write() copies ONE tensor at a time from C-memory
//    (via FFI::string) and immediately writes it to disk.  Peak PHP-heap
//    overhead is O(largest_single_tensor) — typically the embedding table.
//
//    For a 3M-parameter F32 model (≈12 MB), the embedding table is usually
//    the largest tensor (≈2 MB), well inside PHP's 128 MB default heap.
// ═══════════════════════════════════════════════════════════════════════════

final class SafetensorsWriter
{
    /**
     * Serialise named tensors to a .safetensors file.
     *
     * Tensors are written in the iteration order of $tensors (i.e. the order
     * you pass them in = the order they appear in the binary region).
     * HuggingFace tooling re-orders by the JSON header's data_offsets when
     * loading, so insertion order does not affect compatibility.
     *
     * @param string                $path     Destination file path.
     *                                         Parent directory must exist and
     *                                         be writable.
     * @param array<string, Tensor> $tensors  Named tensor map.  All tensors
     *                                         must be dtype=FLOAT32.
     * @param array<string, string> $metadata Optional key-value metadata to
     *                                         embed in the __metadata__ section
     *                                         (e.g. 'step' => '500').
     *
     * @throws \InvalidArgumentException If a tensor is not FLOAT32.
     * @throws \RuntimeException         On I/O failure.
     */
    public static function write(
        string $path,
        array  $tensors,
        array  $metadata = []
    ): void {
        // ── Phase 1: Build the JSON header ────────────────────────────────
        //
        // One forward pass over $tensors to compute cumulative byte offsets.
        // No binary data is read in this phase — only tensor metadata ($size,
        // $shape) is accessed.  This keeps the header-construction cost at
        // O(number_of_tensors), independent of total model size.

        $header = [];

        // The __metadata__ key (if present) must be the FIRST key in the JSON.
        // It maps string keys to string values only (spec requirement).
        if (!empty($metadata)) {
            $header['__metadata__'] = array_map('strval', $metadata);
        }

        // Running byte cursor — position of the NEXT tensor's start byte,
        // measured from the beginning of the binary data region (after the JSON).
        $byteOffset = 0;

        foreach ($tensors as $name => $tensor) {
            if (!$tensor instanceof Tensor) {
                throw new \InvalidArgumentException(
                    "SafetensorsWriter: entry '{$name}' is not a Tensor."
                );
            }

            if ($tensor->dtype !== Tensor::FLOAT32) {
                throw new \InvalidArgumentException(
                    "SafetensorsWriter: tensor '{$name}' has dtype={$tensor->dtype}. "
                    . 'Only FLOAT32 tensors are supported. '
                    . 'Quantise to float32 before saving or use SafetensorsLoader::save().'
                );
            }

            // Float32: 4 bytes per element.  byteLen is the EXCLUSIVE end offset
            // minus the inclusive start offset — i.e. the byte span of this tensor.
            //
            //   byteLen = size * sizeof(float32) = size * 4
            //
            $byteLen = $tensor->size * 4;

            $header[$name] = [
                'dtype'  => 'F32',

                // shape is the logical shape array, e.g. [vocab_size, d_model].
                // SafetensorsLoader reconstructs the same Tensor::$shape from this.
                'shape'  => $tensor->shape,

                // data_offsets: [inclusive_start, exclusive_end] in bytes,
                // measured from byte 0 of the binary region.
                'data_offsets' => [$byteOffset, $byteOffset + $byteLen],
            ];

            // Advance cursor past this tensor's byte block.
            $byteOffset += $byteLen;
        }

        // ── Phase 2: Encode the header to UTF-8 JSON ──────────────────────
        //
        // The spec requires a valid UTF-8 JSON string.  JSON_UNESCAPED_SLASHES
        // prevents escaping forward slashes in tensor names (e.g. layer/weight).
        // JSON_UNESCAPED_UNICODE keeps any non-ASCII metadata values intact.

        $json = json_encode($header, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE);
        if ($json === false) {
            throw new \RuntimeException(
                'SafetensorsWriter: JSON header encoding failed: ' . json_last_error_msg()
            );
        }

        // The 8-byte length field holds the BYTE LENGTH of the JSON string,
        // NOT including the 8 bytes for the length field itself.
        // PHP strlen() returns the byte count for arbitrary binary strings, which
        // is what we want (JSON is ASCII here; no multi-byte chars from tensor names).
        $headerLen = strlen($json);

        // ── Phase 3: Open file and write header + binary data ─────────────

        $fp = fopen($path, 'wb');
        if ($fp === false) {
            throw new \RuntimeException(
                "SafetensorsWriter: cannot open '{$path}' for writing. "
                . 'Check that the directory exists and is writable.'
            );
        }

        try {
            // ── 3a. 8-byte little-endian uint64 header length ─────────────
            //
            // PHP pack 'P' = 64-bit unsigned integer in native byte order
            // (always little-endian on x86/x86-64, which is where PHP runs).
            // The safetensors spec mandates little-endian, so 'P' is correct.
            //
            // Equivalent Python: struct.pack('<Q', header_len)
            fwrite($fp, pack('P', $headerLen));

            // ── 3b. JSON header string ────────────────────────────────────
            fwrite($fp, $json);

            // ── 3c. Binary tensor data — one tensor at a time ─────────────
            //
            // We stream tensors individually to cap peak PHP-heap usage at
            // O(single_largest_tensor) instead of O(total_model_size).
            //
            // FFI::string($buffer, $nBytes) copies $nBytes from C-memory into
            // a PHP string.  This temporary PHP string lives only until fwrite()
            // returns, after which unset() + GC frees it before the next tensor.
            //
            // For a typical embedding table of ~2 MB, this is a single 2 MB
            // PHP string — well within the default 128 MB heap limit.
            //
            // The tensors are written in the same order as their data_offsets
            // declared in the JSON header above — this is a hard requirement.
            // If you change the order here, the offsets will be wrong.

            foreach ($tensors as $name => $tensor) {
                $byteLen = $tensor->size * 4;

                // Copy this tensor's raw float32 bytes from C-memory into a
                // temporary PHP string.  The C buffer is owned by FFI and lives
                // until the Tensor object is garbage-collected.
                $raw = \FFI::string($tensor->buffer, $byteLen);

                fwrite($fp, $raw);

                // Eagerly release the temporary PHP string so the GC can reclaim
                // the heap before we allocate the next tensor's string.
                unset($raw);
            }

        } finally {
            // Always close the file handle, even if an exception is thrown
            // mid-write.  The file may be incomplete on error, but the handle
            // is released so the OS can clean up.
            fclose($fp);
        }
    }

    /**
     * Convenience overload: write a single tensor under a given name.
     *
     * @param string $path     Destination file path.
     * @param string $name     Tensor name in the header.
     * @param Tensor $tensor   The tensor to serialise.
     */
    public static function writeSingle(string $path, string $name, Tensor $tensor): void
    {
        self::write($path, [$name => $tensor]);
    }
}
