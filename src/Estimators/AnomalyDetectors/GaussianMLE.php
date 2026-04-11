<?php

declare(strict_types=1);

namespace Pml\Estimators\AnomalyDetectors;

use Pml\Interfaces\Learner;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Gaussian Maximum Likelihood Estimator (Gaussian MLE).
 * Fits a multivariate Gaussian distribution to the dataset. Anomalies are data points
 * with extremely low Probability Density Function (PDF) evaluations.
 */
final class GaussianMLE implements Learner
{
    private float $threshold; // Contamination threshold logic (e.g. log prob < threshold)
    
    private ?Tensor $mean = null;
    private ?Tensor $variance = null;

    public function __construct(float $threshold = -10.0)
    {
        $this->threshold = $threshold;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        
        $this->mean = $x->meanAxis(0);
        // var = mean( (x - mean)^2 )
        $this->variance = $x->sub($this->mean)->square()->meanAxis(0)->clip(1e-8, INF);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("GaussianMLE is not trained.");

        $x = $dataset->samples();
        
        // Log-Likelihood of Gaussian PDF
        $diffSq = $x->sub($this->mean)->square();
        $term1 = $diffSq->divInplace($this->variance)->sumAxis(1);
        $term2 = $this->variance->mulScalar(2.0 * M_PI)->log()->sum();

        $logProbs = $term1->addScalarInplace($term2)->mulScalarInplace(-0.5);
        
        // 1.0 (Anomaly) if Log-Prob < threshold, else 0.0
        $threshT = Tensor::zeros(1)->addScalarInplace($this->threshold);
        return $logProbs->less($threshT);
    }

    public function trained(): bool
    {
        return $this->mean !== null;
    }
}