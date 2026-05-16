<?php

declare(strict_types=1);

namespace Pml\Interfaces;

/**
 * Opt-in INT8 block quantization for inference-mode layers.
 *
 * After quantize() is called the layer is permanently in inference mode:
 * - forward() runs the quantized kernel (4× less weight memory, faster decode)
 * - backward() throws LogicException (quantized layers are not trainable)
 * - getStateDict() returns dequantized fp32 for checkpoint compatibility
 */
interface Quantizable
{
    /**
     * Convert weight matrices to INT8 symmetric block quantization.
     *
     * @param int $groupSize  Elements per quantization group (32 = Q8_0-class).
     *                        Smaller groups → higher quality, more scale overhead.
     */
    public function quantize(int $groupSize = 32): void;

    public function isQuantized(): bool;
}
