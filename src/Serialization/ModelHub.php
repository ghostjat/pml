<?php

declare(strict_types=1);

namespace Pml\Serialization;

use Pml\Backends\RubixBackend;
use Pml\Backends\TorchBackend;
use Pml\Interfaces\MLBackend;
use RuntimeException;

/**
 * Central registry for saving and loading ML backends.
 *
 * Writes a thin `hub_meta.json` manifest alongside the model artifact
 * so that load() can reconstruct the correct backend without the caller
 * having to know whether the artifact is SafeTensors or RBX.
 *
 * Directory layout:
 *   $dir/hub_meta.json        — backend name + artifact path
 *   $dir/model/               — SafeTensors bundle (TorchBackend)
 *         OR
 *   $dir/model.rbx            — RBX file (RubixBackend)
 *
 * Usage:
 *   ModelHub::save($backend, '/checkpoints/v1');
 *   $backend = ModelHub::load('/checkpoints/v1');
 */
final class ModelHub
{
    private const META_FILE = 'hub_meta.json';

    private function __construct() {}   // static-only class

    public static function save(MLBackend $backend, string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $artifactPath = self::artifactPath($backend, $dir);
        $backend->save($artifactPath);

        $meta = [
            'backend'      => $backend->backendName(),
            'artifact'     => self::relPath($dir, $artifactPath),
            'saved_at'     => date(\DATE_ATOM),
        ];

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . self::META_FILE,
            json_encode($meta, \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );
    }

    public static function load(string $dir): MLBackend
    {
        $metaPath = $dir . \DIRECTORY_SEPARATOR . self::META_FILE;
        if (!is_file($metaPath)) {
            throw new RuntimeException("ModelHub::load — hub_meta.json not found in '$dir'.");
        }

        $meta    = json_decode(file_get_contents($metaPath), true, 512, \JSON_THROW_ON_ERROR);
        $backend = $meta['backend'] ?? '';
        $artifact = $dir . \DIRECTORY_SEPARATOR . $meta['artifact'];

        return match ($backend) {
            'torch' => TorchBackend::load($artifact),
            'rubix' => RubixBackend::load($artifact),
            default => throw new RuntimeException(
                "ModelHub::load — unknown backend '$backend'. "
                . "Register a custom loader or use TorchBackend/RubixBackend::load() directly."
            ),
        };
    }

    // -------------------------------------------------------------------------

    private static function artifactPath(MLBackend $backend, string $dir): string
    {
        return match ($backend->backendName()) {
            'torch' => $dir . \DIRECTORY_SEPARATOR . 'model',      // directory
            'rubix' => $dir . \DIRECTORY_SEPARATOR . 'model.rbx',  // file
            default => $dir . \DIRECTORY_SEPARATOR . 'model',
        };
    }

    private static function relPath(string $base, string $absolute): string
    {
        $base     = rtrim($base, '/\\') . \DIRECTORY_SEPARATOR;
        $absolute = str_replace('\\', '/', $absolute);
        $base     = str_replace('\\', '/', $base);

        if (str_starts_with($absolute, $base)) {
            return substr($absolute, \strlen($base));
        }
        return $absolute;
    }
}
