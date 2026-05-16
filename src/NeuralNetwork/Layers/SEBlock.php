<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * Squeeze-and-Excitation block (Hu et al., 2018).
 *
 * Input  : [N, C, H, W]
 * Output : [N, C, H, W]  (channels rescaled by learned attention weights)
 *
 * Forward:
 *   z = GlobalAvgPool(x)          → [N, C]
 *   z = FC1(z) → ReLU             → [N, C/r]
 *   z = FC2(z) → HardSigmoid      → [N, C]
 *   out = x * reshape(z, [N,C,1,1])
 *
 * All tensor ops executed in C (TensorEngine); no PHP arithmetic.
 */
final class SEBlock implements Layer, Stateful
{
    private Dense      $fc1;
    private ReLU       $relu;
    private Dense      $fc2;
    private HardSigmoid $hSig;

    // Cached for backward
    private ?Tensor $x     = null;
    private ?Tensor $scale = null;   // [N, C, 1, 1]

    public function __construct(int $channels, int $reduction = 4)
    {
        $r         = max(1, (int)($channels / $reduction));
        $this->fc1  = new Dense($channels, $r,   useBias: true);
        $this->relu = new ReLU();
        $this->fc2  = new Dense($r, $channels,   useBias: true);
        $this->hSig = new HardSigmoid();
    }

    public function forward(Tensor $x): Tensor
    {
        $this->x = $x;
        [$N, $C, $H, $W] = $x->shape();

        // Squeeze: global average pool [N,C,H,W] → [N,C]
        $z = $x->meanMulti([2, 3]);

        // Excite
        $z = $this->fc1->forward($z);
        $z = $this->relu->forward($z);
        $z = $this->fc2->forward($z);
        $z = $this->hSig->forward($z);   // [N, C] ∈ [0,1]

        // Reshape to [N, C, 1, 1] for broadcast
        $scale        = $z->reshape($N, $C, 1, 1);
        $ones         = Tensor::ones($N, $C, $H, $W);
        $this->scale  = $scale->mul($ones);   // [N, C, H, W]

        return $x->mul($this->scale);
    }

    public function backward(Tensor $dY): Tensor
    {
        // dX contribution: dY * scale
        $dX = $dY->mul($this->scale);

        // dScale: dY * x → sum over H,W → [N,C]
        $dScale_full = $dY->mul($this->x);
        $dScale_nc   = $dScale_full->meanMulti([2, 3]);

        // Backward through HardSigmoid and FC stack
        $dZ = $this->hSig->backward($dScale_nc);
        $dZ = $this->fc2->backward($dZ);
        $dZ = $this->relu->backward($dZ);
        $dZ = $this->fc1->backward($dZ);

        // dZ is gradient w.r.t. global avg pool output; distribute back to spatial dims
        [$N, $C, $H, $W] = $this->x->shape();
        $inv_hw = 1.0 / ((float)$H * (float)$W);
        $dZ_exp = $dZ->reshape($N, $C, 1, 1)->mul(Tensor::ones($N, $C, $H, $W))->mulScalar($inv_hw);

        return $dX->add($dZ_exp);
    }

    public function getParameters(): array
    {
        return array_merge($this->fc1->getParameters(), $this->fc2->getParameters());
    }

    public function getGradients(): array
    {
        return array_merge($this->fc1->getGradients(), $this->fc2->getGradients());
    }

    public function getStateDict(string $prefix = ''): array
    {
        return array_merge(
            $this->fc1->getStateDict($prefix . 'fc1.'),
            $this->fc2->getStateDict($prefix . 'fc2.')
        );
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->fc1->loadStateDict($dict, $prefix . 'fc1.');
        $this->fc2->loadStateDict($dict, $prefix . 'fc2.');
        $this->x = $this->scale = null;
    }
}
