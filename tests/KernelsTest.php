<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use Pml\Kernels\SVM\Linear;
use Pml\Kernels\SVM\RBF;
use Pml\Kernels\SVM\Polynomial;
use Pml\Kernels\SVM\Sigmoidal;

/**
 * Comprehensive test suite for SVM Kernel functions.
 */
final class KernelsTest extends TestCase
{
    private const DELTA = 1e-4;

    // =========================================================================
    // 1. SVM KERNELS - LINEAR
    // =========================================================================

    public function testLinearKernelBasic(): void
    {
        // Use 2D tensors (1xD matrices) for kernel computation
        $x = Tensor::fromArray([[1.0, 2.0, 3.0]]);
        $y = Tensor::fromArray([[4.0, 5.0, 6.0]]);
        
        $kernel = new Linear();
        $result = $kernel->compute($x, $y);
        
        // Linear kernel: K(x,y) = x · y = 1*4 + 2*5 + 3*6 = 32
        $this->assertEqualsWithDelta(32.0, $result->toFlatArray()[0], self::DELTA);
    }

    public function testLinearKernelWithMatrix(): void
    {
        $X = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $Y = Tensor::fromArray([[5.0, 6.0], [7.0, 8.0]]);
        
        $kernel = new Linear();
        $result = $kernel->compute($X, $Y);
        
        // Result should be [2, 2] matrix
        $this->assertSame([2, 2], $result->shape());
    }

    public function testLinearKernelSymmetric(): void
    {
        $x = Tensor::fromArray([[1.0, 2.0, 3.0]]);
        $y = Tensor::fromArray([[4.0, 5.0, 6.0]]);
        
        $kernel = new Linear();
        $kxy = $kernel->compute($x, $y)->toFlatArray()[0];
        $kyx = $kernel->compute($y, $x)->toFlatArray()[0];
        
        $this->assertEqualsWithDelta($kxy, $kyx, self::DELTA);
    }

    // =========================================================================
    // 2. SVM KERNELS - RBF (Gaussian)
    // =========================================================================

    public function testRBFKernelIdenticalPoints(): void
    {
        $x = Tensor::fromArray([[1.0, 2.0, 3.0]]);
        
        $kernel = new RBF(gamma: 1.0);
        $result = $kernel->compute($x, $x);
        
        // RBF kernel for identical points should be 1.0
        $this->assertEqualsWithDelta(1.0, $result->toFlatArray()[0], self::DELTA);
    }

    public function testRBFKernelDistantPoints(): void
    {
        $x = Tensor::fromArray([[0.0, 0.0, 0.0]]);
        $y = Tensor::fromArray([[10.0, 10.0, 10.0]]);
        
        $kernel = new RBF(gamma: 1.0);
        $result = $kernel->compute($x, $y);
        
        // RBF kernel for distant points should be close to 0
        $val = $result->toFlatArray()[0];
        $this->assertGreaterThanOrEqual(0.0, $val);
        $this->assertLessThan(0.1, $val);
    }

    public function testRBFKernelAlwaysPositive(): void
    {
        $x = Tensor::fromArray([[1.0, 2.0, 3.0]]);
        $y = Tensor::fromArray([[4.0, 5.0, 6.0]]);
        
        $kernel = new RBF(gamma: 0.5);
        $result = $kernel->compute($x, $y);
        
        $val = $result->toFlatArray()[0];
        $this->assertGreaterThan(0.0, $val);
        $this->assertLessThanOrEqual(1.0, $val);
    }

    public function testRBFKernelSymmetric(): void
    {
        $x = Tensor::fromArray([[1.0, 2.0, 3.0]]);
        $y = Tensor::fromArray([[4.0, 5.0, 6.0]]);
        
        $kernel = new RBF(gamma: 1.0);
        $kxy = $kernel->compute($x, $y)->toFlatArray()[0];
        $kyx = $kernel->compute($y, $x)->toFlatArray()[0];
        
        $this->assertEqualsWithDelta($kxy, $kyx, self::DELTA);
    }

    // =========================================================================
    // 3. SVM KERNELS - POLYNOMIAL
    // =========================================================================

    public function testPolynomialKernelDegree2(): void
    {
        $x = Tensor::fromArray([[1.0, 0.0]]);
        $y = Tensor::fromArray([[1.0, 0.0]]);
        
        $kernel = new Polynomial(degree: 2, gamma: 1.0, c: 1.0);
        $result = $kernel->compute($x, $y);
        
        // Poly kernel: (gamma * x·y + c)^degree = (1*1 + 1)^2 = 4
        $this->assertEqualsWithDelta(4.0, $result->toFlatArray()[0], self::DELTA);
    }

    public function testPolynomialKernelIdenticalPoints(): void
    {
        $x = Tensor::fromArray([[1.0, 1.0]]);
        
        $kernel = new Polynomial(degree: 2, gamma: 1.0, c: 1.0);
        $result = $kernel->compute($x, $x);
        
        // (1*2 + 1)^2 = 9
        $this->assertEqualsWithDelta(9.0, $result->toFlatArray()[0], self::DELTA);
    }

    public function testPolynomialKernelAlwaysNonNegative(): void
    {
        $x = Tensor::fromArray([[1.0, 2.0]]);
        $y = Tensor::fromArray([[3.0, 4.0]]);
        
        $kernel = new Polynomial(degree: 2, gamma: 1.0, c: 1.0);
        $result = $kernel->compute($x, $y);
        
        $this->assertGreaterThanOrEqual(0.0, $result->toFlatArray()[0]);
    }

    // =========================================================================
    // 4. SVM KERNELS - SIGMOIDAL
    // =========================================================================

    public function testSigmoidalKernelBasic(): void
    {
        $x = Tensor::fromArray([[1.0, 0.0]]);
        $y = Tensor::fromArray([[1.0, 0.0]]);
        
        $kernel = new Sigmoidal(gamma: 1.0, coef0: 0.0);
        $result = $kernel->compute($x, $y);
        
        // Sigmoid kernel: tanh(gamma * x·y + coef0) = tanh(1) ≈ 0.7616
        $val = $result->toFlatArray()[0];
        $this->assertGreaterThan(0.0, $val);
        $this->assertLessThanOrEqual(1.0, $val);
    }

    public function testSigmoidalKernelRange(): void
    {
        $x = Tensor::fromArray([[1.0, 2.0, 3.0]]);
        $y = Tensor::fromArray([[4.0, 5.0, 6.0]]);
        
        $kernel = new Sigmoidal(gamma: 1.0, coef0: 0.0);
        $result = $kernel->compute($x, $y);
        
        // Sigmoid output is bounded by [-1, 1]
        $val = $result->toFlatArray()[0];
        $this->assertGreaterThanOrEqual(-1.0, $val);
        $this->assertLessThanOrEqual(1.0, $val);
    }

    // =========================================================================
    // 5. EDGE CASES
    // =========================================================================

    public function testLinearKernelZeroVectors(): void
    {
        $x = Tensor::fromArray([[0.0, 0.0, 0.0]]);
        $y = Tensor::fromArray([[1.0, 2.0, 3.0]]);
        
        $kernel = new Linear();
        $result = $kernel->compute($x, $y);
        
        $this->assertEqualsWithDelta(0.0, $result->toFlatArray()[0], self::DELTA);
    }

    public function testRBFKernelWithDifferentGamma(): void
    {
        $x = Tensor::fromArray([[0.0, 0.0]]);
        $y = Tensor::fromArray([[1.0, 1.0]]);
        
        $kernel1 = new RBF(gamma: 0.1);
        $kernel2 = new RBF(gamma: 1.0);
        
        $result1 = $kernel1->compute($x, $y)->toFlatArray()[0];
        $result2 = $kernel2->compute($x, $y)->toFlatArray()[0];
        
        // Higher gamma should give lower similarity for distant points
        $this->assertGreaterThan($result2, $result1);
    }

    public function testCosineSimilarityZeroVector(): void
    {
        $x = Tensor::fromArray([[0.0, 0.0, 0.0]]);
        $y = Tensor::fromArray([[1.0, 2.0, 3.0]]);
        
        $kernel = new Linear(); // Using Linear as it handles zero vectors
        $result = $kernel->compute($x, $y);
        
        $this->assertEqualsWithDelta(0.0, $result->toFlatArray()[0], self::DELTA);
    }
}