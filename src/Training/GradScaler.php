<?php

declare(strict_types=1);

namespace Pml\Training;

use Pml\NeuralNetwork\Layers\Layer;
use Pml\Tensor;

/**
 * Mixed-Precision Gradient Scaler (scaffolding).
 *
 * In a full AMP stack (CUDA + fp16) this class would:
 *   1. Scale the loss before backward() to prevent fp16 underflow.
 *   2. Unscale gradients before the optimizer step.
 *   3. Detect inf/nan gradients and skip the optimizer step if found.
 *   4. Dynamically adjust the scale factor each step.
 *
 * This CPU implementation is a no-op pass-through — it keeps the same
 * public API so that switching to a GPU backend later requires no
 * changes to training loop code.
 *
 * Usage in a training loop:
 *   $scaler = new GradScaler(enabled: false); // CPU
 *   $scaledLoss = $scaler->scale($loss);
 *   $net->backward($scaledLoss);
 *   $scaler->unscaleAndStep($optimizer, $layers);
 *   $scaler->update();
 */
final class GradScaler
{
    private float $scale;
    private int $growthInterval;
    private int $stepsSinceLastInf = 0;

    public function __construct(
        private readonly bool $enabled = false,
        float $initScale = 65536.0,
        private readonly float $growthFactor = 2.0,
        private readonly float $backoffFactor = 0.5,
        int $growthInterval = 2000
    ) {
        $this->scale          = $initScale;
        $this->growthInterval = $growthInterval;
    }

    /**
     * Multiply the loss gradient Tensor by the current scale factor.
     * On CPU (enabled=false) this is a zero-cost pass-through.
     */
    public function scale(Tensor $lossGrad): Tensor
    {
        if (!$this->enabled) {
            return $lossGrad;
        }
        return $lossGrad->mulScalar($this->scale);
    }

    /**
     * Unscale gradients and run the optimizer step.
     * On CPU this simply calls $optimizer->step() directly.
     *
     * @param object    $optimizer  Any optimizer with a step(array) method.
     * @param Layer[]   $layers
     */
    public function unscaleAndStep(object $optimizer, array $layers): void
    {
        if (!$this->enabled) {
            $optimizer->step($layers);
            return;
        }

        // Unscale: divide every gradient by the scale factor.
        $invScale = 1.0 / $this->scale;
        foreach ($layers as $layer) {
            foreach ($layer->getGradients() as $grad) {
                $grad->mulScalarInplace($invScale);
            }
        }

        // Inf/NaN check (placeholder — always clean on CPU).
        $hasInf = false;

        if (!$hasInf) {
            $optimizer->step($layers);
            $this->stepsSinceLastInf++;
        } else {
            $this->scale *= $this->backoffFactor;
            $this->stepsSinceLastInf = 0;
        }
    }

    /**
     * Update scale factor after each optimizer step.
     * On CPU (enabled=false) this is a no-op.
     */
    public function update(): void
    {
        if (!$this->enabled) {
            return;
        }
        if ($this->stepsSinceLastInf >= $this->growthInterval) {
            $this->scale *= $this->growthFactor;
            $this->stepsSinceLastInf = 0;
        }
    }

    public function currentScale(): float { return $this->scale; }

    public function isEnabled(): bool { return $this->enabled; }
}
