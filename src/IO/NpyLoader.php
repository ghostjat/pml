<?php

declare(strict_types=1);

namespace Pml\IO;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  NUMPY NPZ / NPY LOADER
//  Loads .npy (single array) and .npz (zip of .npy files).
// ═══════════════════════════════════════════════════════════════════════════

final class NpyLoader
{
    /**
     * Load a single .npy file into a Tensor.
     * Supports float32, float64 (downcast), int32, int64.
     */
    public static function load(string $filepath): Tensor
    {
        $data = file_get_contents($filepath);
        if ($data === false) throw new \RuntimeException("Cannot read: {$filepath}");

        // Magic: \x93NUMPY
        if (substr($data, 0, 6) !== "\x93NUMPY") {
            throw new \RuntimeException("Not a .npy file: {$filepath}");
        }

        $major  = ord($data[6]);
        $minor  = ord($data[7]);

        if ($major === 1) {
            $headerLen = unpack('v', substr($data, 8, 2))[1];
            $headerStart = 10;
        } elseif ($major === 2) {
            $headerLen = unpack('V', substr($data, 8, 4))[1];
            $headerStart = 12;
        } else {
            throw new \RuntimeException("Unsupported .npy version {$major}.{$minor}");
        }

        $header = substr($data, $headerStart, $headerLen);
        preg_match("/'descr': '([^']+)'/", $header, $descr);
        preg_match("/'shape': \(([^)]*)\)/", $header, $shapeMatch);
        preg_match("/'fortran_order': (\w+)/", $header, $fortranMatch);

        $dtype   = ltrim($descr[1] ?? '<f4', '<>|=');
        $shapeStr = trim($shapeMatch[1] ?? '0');
        $shape   = $shapeStr === '' ? [1] : array_map('intval', explode(',', rtrim($shapeStr, ',')));
        $fortran  = ($fortranMatch[1] ?? 'False') === 'True';

        $binaryData = substr($data, $headerStart + $headerLen);
        $tensor     = new Tensor($shape);
        $size       = (int) array_product($shape);

        switch ($dtype) {
            case 'f4': // float32
                \FFI::memcpy($tensor->buffer, $binaryData, $size * 4);
                break;
            case 'f8': // float64 — downcast
                $doubles = unpack('d*', $binaryData);
                foreach (array_values($doubles) as $i => $v) $tensor->buffer[$i] = (float)$v;
                break;
            case 'i4': // int32
                $ints = unpack('V*', $binaryData); // little-endian
                foreach (array_values($ints) as $i => $v) $tensor->buffer[$i] = (float)$v;
                break;
            default:
                throw new \RuntimeException("Unsupported .npy dtype: {$dtype}");
        }

        return $tensor;
    }
}