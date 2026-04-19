<?php

declare(strict_types=1);

namespace Pml\Interfaces;

/**
 * Explicit persistence contract for classes whose constructor-params or runtime
 * state cannot be reliably inferred by ModelStore's Reflection engine alone.
 *
 * Implement when:
 *   - The class holds FFI\CData properties (e.g. vocab pointers) that Reflection
 *     would silently skip, losing fitted state.
 *   - Constructor param names do NOT match property names.
 *   - You want deterministic, documented serialization without relying on field names.
 *
 * Classes that hold Tensor state should ALSO implement Stateful:
 *   Saveable  → PHP scalars / arrays (including encoded C-state like base64 blobs)
 *   Stateful  → Tensor weights (zero-copy via SafeTensors)
 *
 * ModelStore priority:
 *   Saveable present  → getConfig() / getPhpState()  for PHP-layer state
 *   Saveable absent   → Reflection scan  (automatic fallback for simple classes)
 *   Stateful present  → getStateDict()   for Tensor state (orthogonal to Saveable)
 *   Stateful absent   → Reflection scan  for Tensor-typed properties
 */
interface Saveable
{
    /**
     * Return constructor hyperparameters as a PHP-native array (scalars only).
     * ModelStore passes this to fromConfig() during reconstruction.
     *
     * @return array<string, scalar>
     */
    public function getConfig(): array;

    /**
     * Reconstruct the object from constructor hyperparameters.
     * Must return a fully-initialised (but not-yet-fitted) instance.
     *
     * @param  array<string, scalar> $config
     */
    public static function fromConfig(array $config): static;

    /**
     * Export all runtime state that is NOT covered by Tensor weights.
     * Values must be PHP-native (scalar, null, or plain arrays of same).
     * Never include Tensor, FFI\CData, or resource values here.
     *
     * @return array<string, mixed>
     */
    public function getPhpState(): array;

    /**
     * Restore runtime state previously returned by getPhpState().
     * Called by ModelStore immediately after fromConfig() reconstruction.
     *
     * @param array<string, mixed> $state
     */
    public function setPhpState(array $state): void;
}
