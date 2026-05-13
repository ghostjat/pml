<?php
declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;

final class Gelu implements Layer
{
    private ?Tensor $cachedInput = null;

    public function forward(Tensor $input): Tensor
    {
        $this->cachedInput = $input;
        return $input->gelu();
    }

    public function backward(Tensor $dY): Tensor
    {
        $x = $this->cachedInput;
        if ($x === null) throw new \RuntimeException('Gelu::backward called before forward');
        $this->cachedInput = null;
        return $dY->geluBackward($x);
    }

    public function getParameters(): array { return []; }
    public function getGradients(): array  { return []; }
}
