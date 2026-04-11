<?php
declare(strict_types=1);

namespace Pml;

/**
 * Binary string encoding wrapper — zero-copy when chaining operations.
 */
final class Encoding
{
    private function __construct(private readonly string $data) {}

    public static function wrap(string $data): self
    {
        return new self($data);
    }

    public function data(): string
    {
        return $this->data;
    }

    public function length(): int
    {
        return strlen($this->data);
    }

    public function isBase64(): bool
    {
        return base64_decode($this->data, true) !== false;
    }

    public function toBase64(): self
    {
        return new self(base64_encode($this->data));
    }

    public function fromBase64(): self
    {
        $decoded = base64_decode($this->data, true);
        if ($decoded === false) {
            throw new \RuntimeException("Invalid base64 encoding.");
        }
        return new self($decoded);
    }

    public function compress(): self
    {
        $compressed = gzcompress($this->data, 9);
        if ($compressed === false) {
            throw new \RuntimeException("Failed to compress encoding.");
        }
        return new self($compressed);
    }

    public function decompress(): self
    {
        $decompressed = gzuncompress($this->data);
        if ($decompressed === false) {
            throw new \RuntimeException("Failed to decompress encoding.");
        }
        return new self($decompressed);
    }

    public function __toString(): string
    {
        return $this->data;
    }
}
