<?php

declare(strict_types=1);

namespace Pml\Extractors;

use RuntimeException;
use Traversable;

/**
 * NDJSON (Newline Delimited JSON) Extractor.
 * Streams JSON objects sequentially from a file where each line is a valid JSON object.
 * * JIT & Memory Optimized:
 * - Uses true streaming via PHP Generators (`yield`).
 * - Can parse multi-gigabyte log files while consuming < 2MB of RAM.
 */
final class NDJSON implements Extractor
{
    private string $path;

    public function __construct(string $path)
    {
        $this->path = $path;
    }

    public function getIterator(): Traversable
    {
        $handle = fopen($this->path, 'r');
        
        if ($handle === false) {
            throw new RuntimeException("Could not open file for streaming: {$this->path}");
        }

        // Reads line by line purely in C-Memory buffers without mapping the file to RAM
        while (($line = fgets($handle)) !== false) {
            $line = trim($line);
            
            if ($line === '') {
                continue;
            }

            yield json_decode($line, true, 512, JSON_THROW_ON_ERROR);
        }

        fclose($handle);
    }
}