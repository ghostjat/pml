<?php

declare(strict_types=1);

namespace Pml\Interfaces;

/**
 * Contract for layers and models that own trainable C-memory (Tensor) state.
 *
 * getStateDict() exports a flat, prefixed map of name → Tensor with no PHP copies.
 * loadStateDict() ingests that same map, replacing internal tensors O(1).
 *
 * Prefix convention: "<scope>." — e.g. "layer_0.", "transformer.attn."
 * Flat-key lookup avoids nested traversal and is directly compatible with
 * SafeTensorsIO and HuggingFace weight naming conventions.
 */
interface Stateful
{
    /**
     * Export all trainable parameters as a flat name → Tensor map.
     * Returned Tensors are the live C-memory objects (zero-copy).
     *
     * @param  string $prefix Dot-separated scope prefix prepended to every key.
     * @return array<string, \Pml\Tensor>
     */
    public function getStateDict(string $prefix = ''): array;

    /**
     * Replace internal tensors with tensors from $dict, keyed by "$prefix.$name".
     * Missing keys are silently ignored (supports partial loading / fine-tuning).
     * Implementations MUST perform a zero-copy swap (direct property assignment),
     * never memcpy, to satisfy the O(1) ingestion constraint.
     *
     * @param array<string, \Pml\Tensor> $dict   Flat weight map (e.g. from SafeTensorsIO::load()).
     * @param string                     $prefix Same prefix used during getStateDict().
     */
    public function loadStateDict(array $dict, string $prefix = ''): void;
}
