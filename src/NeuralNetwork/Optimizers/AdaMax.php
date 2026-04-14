<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Optimizers;

use Pml\Tensor;

final class AdaMax implements Optimizer
{
    private float $learningRate;
    private float $beta1;
    private float $beta2;
    private float $epsilon;
    
    private array $mCache = [];
    private array $uCache = [];
    private int $t = 0;

    public function __construct(float $learningRate = 0.002, float $beta1 = 0.9, float $beta2 = 0.999, float $epsilon = 1e-8)
    {
        $this->learningRate = $learningRate;
        $this->beta1 = $beta1;
        $this->beta2 = $beta2;
        $this->epsilon = $epsilon;
    }

    public function step(array $layers): void
    {
        $this->t++;
        $biasCorrection1 = 1.0 - ($this->beta1 ** $this->t);

        foreach ($layers as $layer) {
            foreach ($layer->getParameters() as $name => $paramTensor) {
                $grads = $layer->getGradients();
                
                if (isset($grads[$name])) {
                    $oid = spl_object_id($paramTensor);
                    $g = $grads[$name];

                    if (!isset($this->mCache[$oid])) {
                        $this->mCache[$oid] = Tensor::zeros(...$paramTensor->shape());
                        $this->uCache[$oid] = Tensor::zeros(...$paramTensor->shape());
                    }

                    $m = $this->mCache[$oid];
                    $u = $this->uCache[$oid];

                    // m = beta1 * m + (1 - beta1) * g
                    $m->mulScalarInplace($this->beta1)->addInplace($g->mulScalar(1.0 - $this->beta1));

                    // u = max(beta2 * u, abs(g))
                    $uScaled = $u->mulScalar($this->beta2);
                    $absG = $g->abs();
                    
                    $mask = $uScaled->greater($absG);
                    $uUpdated = $mask->where($uScaled, $absG);
                    $this->uCache[$oid] = $uUpdated;

                    $stepSize = $this->learningRate / $biasCorrection1;
                    
                    // FIXED: Removed Inplace mutation on the cached uUpdated tensor!
                    $denominator = $uUpdated->addScalar($this->epsilon);
                    $update = $m->div($denominator)->mulScalarInplace($stepSize);

                    $paramTensor->subInplace($update);
                }
            }
        }
    }

    public function __sleep(): array { return ['learningRate', 'beta1', 'beta2', 'epsilon', 't']; }
    public function __wakeup(): void { $this->mCache = []; $this->uCache = []; }
}