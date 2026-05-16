<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * Batch Normalisation for 4-D feature maps [N, C, H, W].
 *
 * Normalises over axes {N, H, W} for each channel independently — the
 * standard spatial batch norm used after Conv2D and DepthwiseConv2D.
 *
 * Forward:
 *   mu[c]  = mean of x over {N, H, W}          (computed via C multi-axis reduce)
 *   var[c] = variance of x over {N, H, W}
 *   x_hat  = (x - mu) / sqrt(var + eps)
 *   y      = gamma[c] * x_hat + beta[c]
 *
 * gamma and beta are [1, C, 1, 1] for broadcasting against [N, C, H, W].
 *
 * Zero PHP arithmetic — all ops via TensorEngine C.
 */
final class BatchNorm2D implements Layer, Stateful, HasTrainingMode
{
    private Tensor $gamma;           // [1, C, 1, 1]
    private Tensor $beta;            // [1, C, 1, 1]
    private Tensor $runningMean;     // [1, C, 1, 1]
    private Tensor $runningVar;      // [1, C, 1, 1]

    private ?Tensor $xNorm  = null;
    private ?Tensor $std    = null;
    private ?array  $shape  = null;
    private ?Tensor $dGamma = null;
    private ?Tensor $dBeta  = null;

    public bool $training = true;

    public function __construct(
        private readonly int   $channels,
        private readonly float $momentum = 0.9,
        private readonly float $eps      = 1e-5,
    ) {
        $this->gamma       = Tensor::ones(1, $channels, 1, 1);
        $this->beta        = Tensor::zeros(1, $channels, 1, 1);
        $this->runningMean = Tensor::zeros(1, $channels, 1, 1);
        $this->runningVar  = Tensor::ones(1, $channels, 1, 1);
    }

    public function forward(Tensor $input): Tensor
    {
        $this->shape = $input->shape();   // [N, C, H, W]
        [$N, $C, $H, $W] = $this->shape;

        if ($this->training) {
            // Compute mean and variance over {N, H, W} per channel → [1, C, 1, 1]
            $mu  = $input->meanMulti([0, 2, 3])->reshape(1, $C, 1, 1);
            $xC  = $input->sub($mu);
            $var = $xC->square()->meanMulti([0, 2, 3])->reshape(1, $C, 1, 1);

            // Update running stats
            $this->runningMean->mulScalarInplace($this->momentum)
                ->addInplace($mu->mulScalar(1.0 - $this->momentum));
            $this->runningVar->mulScalarInplace($this->momentum)
                ->addInplace($var->mulScalar(1.0 - $this->momentum));

            $this->std   = $var->addScalarInplace($this->eps)->sqrt();
            $this->xNorm = $xC->divInplace($this->std);
        } else {
            $std = $this->runningVar->addScalar($this->eps)->sqrt();
            $this->xNorm = $input->sub($this->runningMean)->divInplace($std);
        }

        return $this->xNorm->mul($this->gamma)->addInplace($this->beta);
    }

    public function backward(Tensor $dY): Tensor
    {
        [$N, $C, $H, $W] = $this->shape;
        $m = (float)($N * $H * $W);

        $this->dGamma = $dY->mul($this->xNorm)->sumMulti([0, 2, 3])->reshape(1, $C, 1, 1);
        $this->dBeta  = $dY->sumMulti([0, 2, 3])->reshape(1, $C, 1, 1);

        $dXn     = $dY->mul($this->gamma);
        $dVar    = $dXn->mul($this->xNorm)->mulScalar(-0.5)
                       ->sumMulti([0, 2, 3])->reshape(1, $C, 1, 1)
                       ->mul($this->std->pow(Tensor::fill([-3.0], 1, $C, 1, 1)));
        $dMu1    = $dXn->mul(Tensor::ones($N, $C, $H, $W)->mulScalar(-1.0 / $m)->div($this->std));
        $dMu2    = $dVar->mul($this->xNorm->mulScalar(-2.0 / $m));
        $dMu     = $dMu1->sumMulti([0, 2, 3])->reshape(1, $C, 1, 1)
                        ->add($dMu2->sumMulti([0, 2, 3])->reshape(1, $C, 1, 1));

        return $dXn->div($this->std)
                   ->add($dVar->mul($this->xNorm->mulScalar(2.0 / $m)))
                   ->add($dMu->mulScalar(1.0 / $m));
    }

    public function getParameters(): array
    {
        return ['gamma' => $this->gamma, 'beta' => $this->beta];
    }

    public function getGradients(): array
    {
        return ['gamma' => $this->dGamma, 'beta' => $this->dBeta];
    }

    public function getStateDict(string $prefix = ''): array
    {
        return [
            $prefix . 'gamma'       => $this->gamma,
            $prefix . 'beta'        => $this->beta,
            $prefix . 'runningMean' => $this->runningMean,
            $prefix . 'runningVar'  => $this->runningVar,
        ];
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->gamma       = $dict[$prefix . 'gamma'];
        $this->beta        = $dict[$prefix . 'beta'];
        $this->runningMean = $dict[$prefix . 'runningMean'] ?? Tensor::zeros(1, $this->channels, 1, 1);
        $this->runningVar  = $dict[$prefix . 'runningVar']  ?? Tensor::ones(1, $this->channels, 1, 1);
        $this->xNorm = $this->std = $this->dGamma = $this->dBeta = $this->shape = null;
    }

    public function setTraining(bool $mode): void
    {
        $this->training = $mode;
    }
}
