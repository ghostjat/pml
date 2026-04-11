<?php

declare(strict_types=1);

namespace Pml\Interfaces;

/**
 * Interface for saving and loading the internal state of a model.
 * Bypasses PHP serialize() entirely to prevent memory leaks, relying on direct C binary dumps.
 */
interface Persistable
{
    /**
     * Save the C-struct weights and biases to a binary file.
     */
    public function save(string $filepath): void;

    /**
     * Reconstruct the model and its FFI pointers from a binary file.
     * * @param string $filepath
     * @return self
     */
    public static function load(string $filepath): self;
}