<?php

declare(strict_types=1);

namespace Pml\Vision;

/**
 * VideoCapture — stub.
 *
 * Video decoding requires a video I/O backend (e.g. libav / FFmpeg).
 * This stub reserves the namespace so future extensions can be added
 * without breaking the existing PHP API.
 */
final class VideoCapture
{
    public function __construct(string $path)
    {
        throw new \RuntimeException(
            'VideoCapture is not yet implemented. '
            . 'Video I/O requires a video backend (FFmpeg/libav). '
            . 'Use pre-extracted frame images with Image::read() instead.'
        );
    }
}
