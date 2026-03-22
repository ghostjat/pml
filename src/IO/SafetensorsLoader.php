<?php

declare(strict_types=1);

namespace Pml\IO;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  SAFETENSORS LOADER
//  Zero-parse direct-to-C-memory load from HuggingFace .safetensors format.
// ═══════════════════════════════════════════════════════════════════════════

final class SafetensorsLoader
{
    /**
     * Load all tensors from a .safetensors file.
     *
     * Format:
     *   [8 bytes: uint64 LE header_len] [header_len bytes: JSON] [binary data]
     *
     * Supports F32 (native), BF16 and F16 with automatic upcast to F32.
     *
     * @return array<string, Tensor>
     */
    public static function load(string $filepath, bool $verbose = false): array
    {
        if (!file_exists($filepath)) {
            throw new \RuntimeException("File not found: {$filepath}");
        }

        $fp = fopen($filepath, 'rb');
        if (!$fp) throw new \RuntimeException("Cannot open: {$filepath}");

        try {
            // ── Header ──────────────────────────────────────────────────────
            $headerBytes  = fread($fp, 8);
            $headerLength = unpack('P', $headerBytes)[1]; // uint64 little-endian

            $jsonBytes  = fread($fp, $headerLength);
            $metadata   = json_decode($jsonBytes, true, flags: JSON_THROW_ON_ERROR);

            $dataOffset = 8 + $headerLength; // binary data begins here

            $tensors = [];

            foreach ($metadata as $name => $info) {
                if ($name === '__metadata__') continue;

                $dtype   = $info['dtype'];
                $shape   = $info['shape'];
                $offsets = $info['data_offsets'];
                $byteLen = $offsets[1] - $offsets[0];

                fseek($fp, $dataOffset + $offsets[0]);
                $raw = fread($fp, $byteLen);

                $tensor = match ($dtype) {
                    'F32'  => self::loadF32($raw, $shape),
                    'F16'  => self::loadF16($raw, $shape),
                    'BF16' => self::loadBF16($raw, $shape),
                    'I32'  => self::loadI32AsF32($raw, $shape),
                    default => throw new \RuntimeException(
                        "Unsupported dtype '{$dtype}' for tensor '{$name}'. "
                        . "PhpTensor supports F32, F16, BF16, I32."
                    ),
                };

                $tensors[$name] = $tensor;

                if ($verbose) {
                    $shapeStr = implode('×', $shape);
                    echo "[SafetensorsLoader] Loaded '{$name}' ({$dtype}) [{$shapeStr}]\n";
                }
            }

            return $tensors;

        } finally {
            fclose($fp);
        }
    }

    /**
     * Load only specific tensor names (lazy partial load).
     * Useful when you only need a subset of a large checkpoint.
     *
     * @param string[] $names
     * @return array<string, Tensor>
     */
    public static function loadKeys(string $filepath, array $names): array
    {
        $all    = self::load($filepath);
        $result = [];
        foreach ($names as $name) {
            if (!isset($all[$name])) {
                throw new \RuntimeException("Tensor '{$name}' not found in {$filepath}.");
            }
            $result[$name] = $all[$name];
        }
        return $result;
    }

    /**
     * Save tensors to .safetensors format.
     *
     * @param array<string, Tensor> $tensors
     */
    public static function save(string $filepath, array $tensors): void
    {
        $metadata   = [];
        $binaryParts = [];
        $offset     = 0;

        foreach ($tensors as $name => $tensor) {
            $byteLen         = $tensor->size * 4;
            $metadata[$name] = [
                'dtype'        => 'F32',
                'shape'        => $tensor->shape,
                'data_offsets' => [$offset, $offset + $byteLen],
            ];
            $binaryParts[$name] = \FFI::string($tensor->buffer, $byteLen);
            $offset += $byteLen;
        }

        $json       = json_encode($metadata, JSON_UNESCAPED_SLASHES);
        $headerLen  = strlen($json);
        $headerLenPacked = pack('P', $headerLen);

        $fp = fopen($filepath, 'wb');
        if (!$fp) throw new \RuntimeException("Cannot write to: {$filepath}");

        fwrite($fp, $headerLenPacked);
        fwrite($fp, $json);
        foreach ($binaryParts as $chunk) fwrite($fp, $chunk);
        fclose($fp);
    }

    // ── Private dtype handlers ─────────────────────────────────────────────

    private static function loadF32(string $raw, array $shape): Tensor
    {
        $tensor = new Tensor($shape);
        \FFI::memcpy($tensor->buffer, $raw, strlen($raw));
        return $tensor;
    }

    private static function loadF16(string $raw, array $shape): Tensor
    {
        $size   = (int) array_product($shape);
        $tensor = new Tensor($shape);
        $halfs  = unpack('v*', $raw); // unsigned 16-bit LE

        foreach (array_values($halfs) as $i => $h) {
            $tensor->buffer[$i] = self::f16ToF32($h);
        }
        return $tensor;
    }

    private static function loadBF16(string $raw, array $shape): Tensor
    {
        $size   = (int) array_product($shape);
        $tensor = new Tensor($shape);
        $halfs  = unpack('v*', $raw);

        foreach (array_values($halfs) as $i => $h) {
            // BF16 → F32: just zero-pad the mantissa (shift left 16 bits)
            $f32bits = $h << 16;
            $tensor->buffer[$i] = unpack('f', pack('V', $f32bits))[1];
        }
        return $tensor;
    }

    private static function loadI32AsF32(string $raw, array $shape): Tensor
    {
        $tensor = new Tensor($shape);
        $ints   = unpack('V*', $raw); // uint32 LE
        foreach (array_values($ints) as $i => $v) {
            $tensor->buffer[$i] = (float)$v;
        }
        return $tensor;
    }

    /** IEEE 754 float16 → float32 conversion */
    private static function f16ToF32(int $h): float
    {
        $sign = ($h >> 15) & 0x1;
        $exp  = ($h >> 10) & 0x1F;
        $mant = $h & 0x3FF;

        if ($exp === 0) {
            if ($mant === 0) return $sign ? -0.0 : 0.0;
            // Subnormal
            $exp32 = 127 - 14;
            while (!($mant & 0x400)) { $mant <<= 1; $exp32--; }
            $mant &= 0x3FF;
        } elseif ($exp === 31) {
            $f32bits = ($sign << 31) | 0x7F800000 | ($mant << 13);
            return unpack('f', pack('V', $f32bits))[1];
        } else {
            $exp32 = $exp + (127 - 15);
        }

        $f32bits = ($sign << 31) | ($exp32 << 23) | ($mant << 13);
        return unpack('f', pack('V', $f32bits))[1];
    }
}

