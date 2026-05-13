<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

/**
 * Global Average Pooling for 2D feature maps.
 *
 * Input : [B, C, H, W]
 * Output: [B, C]  (spatial average over H×W per channel)
 *
 * Replaces Flatten + large Dense head — dramatically reduces parameter count
 * and adds spatial invariance.  Backward pass redistributes gradients
 * uniformly across all spatial positions (grad = dY / (H*W)).
 */
final class GlobalAveragePooling2D implements Layer
{
    private ?array $inputShape = null;

    public function forward(Tensor $input): Tensor
    {
        $s = $input->shape();
        if (count($s) !== 4) {
            throw new \RuntimeException(
                sprintf('[GlobalAveragePooling2D] expected 4-D input [B,C,H,W], got [%s]', implode(',', $s))
            );
        }
        $this->inputShape = $s;
        // Mean over H (axis 2) and W (axis 3) → [B, C]
        return $input->meanMulti([2, 3]);
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->inputShape === null) {
            throw new \RuntimeException('[GlobalAveragePooling2D] backward called before forward');
        }
        [, , $H, $W] = $this->inputShape;
        $scale = 1.0 / ($H * $W);
        // dY is [B, C]; expand back to [B, C, H, W] by broadcasting
        // Reshape to [B, C, 1, 1] then expand (multiply by ones [1,1,H,W])
        [$B, $C] = $this->inputShape;
        $expanded = $dY->reshape($B, $C, 1, 1);
        $ones     = Tensor::ones($B, $C, $H, $W);
        return $expanded->mul($ones)->mulScalar($scale);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array  { return []; }
}
