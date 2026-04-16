<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;
use RuntimeException;

/**
 * Batch Normalization Layer.
 * Stabilizes and accelerates deep network training by standardizing feature activations.
 * * JIT & Memory Optimized:
 * - Evaluates Complex Jacobian gradients using 100% C-level axis reductions and broadcasting.
 * - Safely manages Inference Moving Averages using zero-allocation In-Place mutators.
 */
final class BatchNormalization implements Layer, Stateful, HasTrainingMode
{
    private int $features;
    private float $momentum;
    private float $eps;
    
    // Learnable Parameters
    private Tensor $gamma;
    private Tensor $beta;
    
    // Inference States (Exponential Moving Averages)
    private Tensor $runningMean;
    private Tensor $runningVar;

    // Cached states for Backpropagation
    private ?Tensor $xNorm = null;
    private ?Tensor $std = null;
    private ?array $inputShape = null;
    
    private ?Tensor $dGamma = null;
    private ?Tensor $dBeta = null;

    /** @var bool Flag to toggle behavior between train and evaluate modes */
    public bool $training = true;

    public function __construct(int $features, float $momentum = 0.9, float $eps = 1e-5)
    {
        $this->features = $features;
        $this->momentum = $momentum;
        $this->eps = $eps;

        // Initialize Scale (gamma) to 1, and Shift (beta) to 0
        $this->gamma = Tensor::ones(1, $features);
        $this->beta = Tensor::zeros(1, $features);

        // Initialize Moving Averages for Inference mode
        $this->runningMean = Tensor::zeros(1, $features);
        $this->runningVar = Tensor::ones(1, $features);
    }

    public function forward(Tensor $input): Tensor
    {
        $this->inputShape = $input->shape();

        if ($this->training) {
            // --- TRAINING MODE ---

            // 1. Calculate Batch Mean and Variance across Axis 0
            $mu = $input->meanAxis(0);
            $xCentered = $input->sub($mu); // Zero-alloc broadcasting in C
            
            // var = mean( (X - mu)^2 )
            $var = $xCentered->square()->meanAxis(0);

            // 2. Update Running Statistics for Inference
            // Uses In-Place mutations sequentially to prevent Garbage Collection spikes
            $this->runningMean->mulScalarInplace($this->momentum)
                 ->addInplace($mu->mulScalar(1.0 - $this->momentum));
                 
            $this->runningVar->mulScalarInplace($this->momentum)
                 ->addInplace($var->mulScalar(1.0 - $this->momentum));

            // 3. Normalize the batch
            $this->std = $var->addScalarInplace($this->eps)->sqrt();
            $this->xNorm = $xCentered->divInplace($this->std);

            // 4. Scale and Shift: Y = gamma * X_norm + beta
            return $this->xNorm->mul($this->gamma)->addInplace($this->beta);
            
        } else {
            // --- INFERENCE MODE ---
            
            // Freezes gradients and uses the learned moving averages
            $std = $this->runningVar->addScalar($this->eps)->sqrt();
            $xNorm = $input->sub($this->runningMean)->divInplace($std);
            
            return $xNorm->mulInplace($this->gamma)->addInplace($this->beta);
        }
    }

    public function backward(Tensor $dY): Tensor
    {
        if (!$this->training) {
            throw new RuntimeException("Backward pass called during inference mode.");
        }
        if ($this->xNorm === null || $this->std === null) {
            throw new RuntimeException("Backward called before forward pass.");
        }

        $N = (float) $this->inputShape[0]; // Batch Size

        // 1. Compute Gradients w.r.t. Learnable Parameters
        $this->dGamma = $dY->mul($this->xNorm)->sumAxis(0);
        $this->dBeta = $dY->sumAxis(0);

        // 2. Compute Gradient w.r.t. Input (dX) using the vectorized Jacobian
        // dX = (gamma / (N * std)) * (N * dY - sum(dY) - X_norm * sum(dY * X_norm))
        
        $NdY = $dY->mulScalar($N);
        $term2 = $this->xNorm->mul($this->dGamma);
        
        // Compute inner bracket: N * dY - dBeta - X_norm * dGamma
        $dX = $NdY->subInplace($this->dBeta)->subInplace($term2);
        
        // Compute scalar multiplier and apply
        $scale = $this->gamma->div($this->std->mulScalar($N));
        
        return $dX->mulInplace($scale);
    }

    public function getParameters(): array
    {
        // Explicitly track running averages so they get serialized to SSD
        return [
            'gamma'       => $this->gamma,
            'beta'        => $this->beta,
            'runningMean' => $this->runningMean,
            'runningVar'  => $this->runningVar
        ];
    }

    public function getGradients(): array
    {
        return [
            'gamma' => $this->dGamma,
            'beta'  => $this->dBeta
        ];
    }

    public function getConfig(): array
    {
        return [
            'features' => $this->features,
            'momentum' => $this->momentum,
            'eps'      => $this->eps,
        ];
    }

    public static function fromConfig(array $config): static
    {
        return new static(
            (int)   $config['features'],
            (float) $config['momentum'],
            (float) $config['eps']
        );
    }

    public function getStateDict(string $prefix = ''): array
    {
        return [
            "{$prefix}gamma"       => $this->gamma,
            "{$prefix}beta"        => $this->beta,
            "{$prefix}runningMean" => $this->runningMean,
            "{$prefix}runningVar"  => $this->runningVar,
        ];
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->gamma       = $dict["{$prefix}gamma"];
        $this->beta        = $dict["{$prefix}beta"];
        $this->runningMean = $dict["{$prefix}runningMean"];
        $this->runningVar  = $dict["{$prefix}runningVar"];
        $this->xNorm       = null;
        $this->std         = null;
        $this->dGamma      = null;
        $this->dBeta       = null;
    }

    public function setTraining(bool $mode): void
    {
        $this->training = $mode;
    }
}