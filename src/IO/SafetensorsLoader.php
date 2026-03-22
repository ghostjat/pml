<?php
declare(strict_types=1);

namespace Pml\IO;

use Pml\Tensor;

class SafetensorsLoader
{
    /**
     * Parses a .safetensors file and loads weights directly into FFI memory.
     * @return array<string, Tensor>
     */
    public static function load(string $filepath): array
    {
        $fp = fopen($filepath, 'rb');
        if (!$fp) throw new \RuntimeException("Cannot open file: $filepath");

        // 1. Read the first 8 bytes (Little-Endian UInt64) to get JSON header length
        $headerBytes = fread($fp, 8);
        $headerLength = unpack('P', $headerBytes)[1];

        // 2. Read and decode the JSON metadata
        $jsonBytes = fread($fp, $headerLength);
        $metadata = json_decode($jsonBytes, true);
        
        // The binary data starts exactly here
        $dataStartOffset = 8 + $headerLength;
        
        $tensors = [];
        
        // 3. Extract each tensor directly into C memory
        foreach ($metadata as $tensorName => $info) {
            if ($tensorName === '__metadata__') continue;
            
            // Only support Float32 (F32) for this architecture
            if ($info['dtype'] !== 'F32') {
                throw new \RuntimeException("Unsupported dtype: {$info['dtype']} for tensor {$tensorName}. Expected F32.");
            }

            $shape = $info['shape'];
            $offsets = $info['data_offsets'];
            $byteSize = $offsets[1] - $offsets[0];
            
            // Seek to the exact byte location of this tensor's data
            fseek($fp, $dataStartOffset + $offsets[0]);
            $rawData = fread($fp, $byteSize);
            
            // Create the Tensor (allocates C memory)
            $tensor = new Tensor($shape);
            
            // Violent, zero-parse memory copy straight to the C buffer
            \FFI::memcpy($tensor->buffer, $rawData, $byteSize);
            
            $tensors[$tensorName] = $tensor;
        }

        fclose($fp);
        return $tensors;
    }
}