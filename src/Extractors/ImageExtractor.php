<?php

declare(strict_types=1);

namespace Pml\Extractors;

use Traversable;
use RuntimeException;
use RecursiveDirectoryIterator;
use RecursiveIteratorIterator;

/**
 * Image Extractor.
 * Streams images from a nested folder directory where the folder name serves as the class label.
 * (e.g., `data/cats/img1.jpg`, `data/dogs/img2.jpg`)
 * * JIT & Memory Optimized:
 * - Uses PHP Generators to stream metadata without loading thousands of images into RAM.
 * - Decodes using native C-compiled GD extensions into contiguous flat RGB matrices.
 */
final class ImageExtractor implements Extractor
{
    private string $directory;

    public function __construct(string $directory)
    {
        if (!is_dir($directory)) {
            throw new RuntimeException("Directory not found: {$directory}");
        }
        $this->directory = $directory;
    }

    public function getIterator(): Traversable
    {
        $iterator = new RecursiveIteratorIterator(new RecursiveDirectoryIterator($this->directory));

        foreach ($iterator as $file) {
            if ($file->isFile() && in_array(strtolower($file->getExtension()), ['jpg', 'jpeg', 'png'])) {
                
                $label = basename(dirname($file->getPathname()));
                $imagePath = $file->getPathname();

                // Decode Image to GD Resource natively in PHP's C-Space
                $img = @imagecreatefromstring(file_get_contents($imagePath));
                if (!$img) continue;

                $width = imagesx($img);
                $height = imagesy($img);
                
                // Extract to a flat RGB array [R, G, B, R, G, B...]
                $pixels = [];
                for ($y = 0; $y < $height; $y++) {
                    for ($x = 0; $x < $width; $x++) {
                        $rgb = imagecolorat($img, $x, $y);
                        $pixels[] = ($rgb >> 16) & 0xFF; // R
                        $pixels[] = ($rgb >> 8) & 0xFF;  // G
                        $pixels[] = $rgb & 0xFF;         // B
                    }
                }
                
                imagedestroy($img);

                // Yield the raw RGB flat array and the string label
                yield ['pixels' => $pixels, 'label' => $label, 'width' => $width, 'height' => $height];
            }
        }
    }
}