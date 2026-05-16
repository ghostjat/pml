<?php

declare(strict_types=1);

namespace Pml\Estimators\AnomalyDetectors;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Kernels\SVM\Kernel;
use Pml\Kernels\SVM\RBF;
use Pml\Kernels\SVM\Linear;
use Pml\Kernels\SVM\Polynomial;
use RuntimeException;

/**
 * One-Class Support Vector Machine.
 * Unsupervised outlier detection that learns a boundary maximizing the margin from the origin 
 * mapped into a high-dimensional kernel space.
 */
final class OneClassSVM implements Learner, Persistable
{
    private float $nu;
    private Kernel $kernel;
    private int $epochs;
    private float $learningRate;
    
    private ?Tensor $weights = null; // Dual alpha coefficients
    private float $rho = 0.0;
    private ?Tensor $supportVectors = null;

    public function __construct(float $nu = 0.1, ?Kernel $kernel = null, int $epochs = 100, float $learningRate = 0.01)
    {
        $this->nu = $nu;
        $this->kernel = $kernel ?? new RBF(0.1);
        $this->epochs = $epochs;
        $this->learningRate = $learningRate;
    }

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $x = $dataset->samples();
        $n = (float) $x->shape()[0];
        
        $this->supportVectors = $x;
        // Alphas bounded by [0, 1 / (nu * N)]
        $this->weights = Tensor::randomUniform([$x->shape()[0], 1], 0.0, 1.0 / ($this->nu * $n));
        $this->rho = 1.0;

        $kMatrix = $this->kernel->compute($x, $this->supportVectors);

        for ($e = 0; $e < $this->epochs; $e++) {
            // Z = K * alpha
            $z = $kMatrix->matmul($this->weights);
            
            // Subgradient mask where (Z < rho)
            $rhoT = Tensor::zeros(1)->addScalarInplace($this->rho);
            $violationMask = $z->less($rhoT);

            // Update Alphas
            $kMatrixT = $kMatrix->transpose();
            $dZ = $violationMask->mulScalar(-1.0);
            
            // dAlpha = Alpha + (1 / (nu * N)) * K^T * dZ
            $dw = $kMatrixT->matmul($dZ)->mulScalarInplace(1.0 / ($this->nu * $n))->addInplace($this->weights);
            
            $this->weights->subInplace($dw->mulScalarInplace($this->learningRate));
            
            // Bound alphas: 0 <= alpha <= 1 / (nu * N)
            $this->weights = $this->weights->clip(0.0, 1.0 / ($this->nu * $n));
            
            // Update Rho: rho -= lr * (-1 + sum(violationMask) / (nu * N))
            $dRho = -1.0 + ($violationMask->sum() / ($this->nu * $n));
            $this->rho -= $this->learningRate * $dRho;
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("OneClassSVM is not trained.");

        $kTest = $this->kernel->compute($dataset->samples(), $this->supportVectors);
        $z = $kTest->matmul($this->weights);
        
        // Anomalies are instances where Z < rho
        $rhoT = Tensor::zeros(1)->addScalarInplace($this->rho);
        return $z->less($rhoT);
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        $kernelClass = get_class($this->kernel);
        $kernelParams = [];
        $ref = new \ReflectionObject($this->kernel);
        foreach ($ref->getProperties() as $prop) {
            $prop->setAccessible(true);
            $kernelParams[$prop->getName()] = $prop->getValue($this->kernel);
        }
        file_put_contents($dir . '/config.json', json_encode(['nu' => $this->nu, 'epochs' => $this->epochs, 'learningRate' => $this->learningRate, 'rho' => $this->rho, 'kernelClass' => $kernelClass, 'kernelParams' => $kernelParams]));
        if ($this->weights !== null) {
            SafeTensorsIO::save($dir . '/model.safetensors', ['weights' => $this->weights, 'support_vectors' => $this->supportVectors]);
        }
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $p = $c['kernelParams'] ?? [];
        $kernel = match ($c['kernelClass']) {
            RBF::class        => new RBF((float) ($p['gamma'] ?? 0.1)),
            Polynomial::class => new Polynomial((int) ($p['degree'] ?? 3), (float) ($p['gamma'] ?? 1.0), (float) ($p['c'] ?? 1.0)),
            default           => new Linear(),
        };
        $i = new self((float) $c['nu'], $kernel, (int) $c['epochs'], (float) $c['learningRate']);
        $i->rho = (float) $c['rho'];
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $t = SafeTensorsIO::load($stPath);
            $i->weights = $t['weights'] ?? null;
            $i->supportVectors = $t['support_vectors'] ?? null;
        }
        return $i;
    }
}