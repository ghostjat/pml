<?php

declare(strict_types=1);

namespace Pml\Kernels\Distance;

use Pml\Tensor;
use InvalidArgumentException;

/**
 * Mahalanobis Distance.
 * Measures distance relative to a centroid, accounting for the correlation of the dataset.
 */
final class Mahalanobis implements Distance
{
    private Tensor $inverseCovariance;

    /**
     * @param Tensor $inverseCovariance The inverted covariance matrix of the dataset [D, D].
     */
    public function __construct(Tensor $inverseCovariance)
    {
        $this->inverseCovariance = $inverseCovariance;
    }

    public function compute(Tensor $a, Tensor $b): Tensor
    {
        // Distance = sqrt( (B - A) * V_inv * (B - A)^T )
        $diff = $b->sub($a); // Shape: [N, D]
        
        // Transform the difference by the inverse covariance
        $transformed = $diff->matmul($this->inverseCovariance); // Shape: [N, D]
        
        // Dot product with itself along the feature axis
        // Equivalent to diag(transformed * diff^T)
        $dot = $transformed->mul($diff)->sumAxis(1);
        
        return $dot->sqrt();
    }
}