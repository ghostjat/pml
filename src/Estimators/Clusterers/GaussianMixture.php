<?php

declare(strict_types=1);

namespace Pml\Estimators\Clusterers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Clusterers\Seeders\PlusPlus;
use RuntimeException;

/**
 * Gaussian Mixture Model (GMM) - Diagonal Covariance.
 * A probabilistic model that assumes data is generated from a mixture of K Gaussian distributions.
 * * JIT & Memory Optimized:
 * - Calculates probabilities safely without floating-point overflow via the `Log-Sum-Exp` hardware trick.
 * - Covariances and Means update concurrently via zero-copy subset slicing and OpenBLAS matrix broadcasting.
 */
final class GaussianMixture implements Learner, Probabilistic, Persistable
{
    private int $k;
    private int $maxIter;
    private float $tolerance;

    private ?Tensor $means = null;
    private ?Tensor $vars = null;
    private array $priors = [];

    public function __construct(int $k = 3, int $maxIter = 100, float $tolerance = 1e-4)
    {
        $this->k = $k;
        $this->maxIter = $maxIter;
        $this->tolerance = $tolerance;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = (float) $x->shape()[0];
        $d = $x->shape()[1];

        // 1. Initialize using KMeans++ logic
        $seeder = new PlusPlus();
        $this->means = $seeder->seed($dataset, $this->k);

        // 2. Initialize Variance uniformly, and Priors to 1/K
        $varGlobal = $x->variance();
        $this->vars = Tensor::ones($this->k, $d)->mulScalarInplace($varGlobal);
        $this->priors = array_fill(0, $this->k, 1.0 / $this->k);

        $prevLl = -INF;

        for ($iter = 0; $iter < $this->maxIter; $iter++) {
            
            // --- EXPECTATION (E-Step) ---
            $logProbsList = [];
            for ($c = 0; $c < $this->k; $c++) {
                $mean = $this->means->row($c);
                $var = $this->vars->row($c);

                // log_prob = -0.5 * sum( (x - mu)^2 / var + log(2*pi*var), 1 ) + log(prior)
                $diffSq = $x->sub($mean)->square();
                $term1 = $diffSq->divInplace($var)->sumAxis(1);
                $term2 = $var->mulScalar(2.0 * M_PI)->log()->sum();

                $logProb = $term1->addScalarInplace($term2)
                                 ->mulScalarInplace(-0.5)
                                 ->addScalarInplace(log($this->priors[$c]));
                                 
                $logProbsList[] = $logProb->expandDims(1);
            }
            $logProbs = Tensor::concat($logProbsList, 1); // Shape [N, K]

            // Log-Sum-Exp Trick: LSE = max + log(sum(exp(x - max)))
            $maxLogProb = $logProbs->maxAxis(1)->expandDims(1);
            $sumExp = $logProbs->sub($maxLogProb)->exp()->sumAxis(1)->expandDims(1);
            $lse = $maxLogProb->addInplace($sumExp->log()); // Shape [N, 1]

            // Responsibilities (gamma) = exp(logProbs - LSE)
            $gamma = $logProbs->sub($lse)->exp(); // Shape [N, K]

            // --- MAXIMIZATION (M-Step) ---
            $N_c = $gamma->sumAxis(0); // Sum over N -> Shape [K]
            $N_c_flat = $N_c->toFlatArray();
            $gamma_T = $gamma->transpose(); // Shape [K, N]

            // Update Means = (gamma_T * X) / N_c
            $N_c_expanded = $N_c->expandDims(1)->addScalarInplace(1e-8);
            $this->means = $gamma_T->matmul($x)->divInplace($N_c_expanded);

            // Update Variances
            $newVarsList = [];
            for ($c = 0; $c < $this->k; $c++) {
                $mean = $this->means->row($c);
                $gamma_c = $gamma_T->row($c)->expandDims(1); // Shape [N, 1]
                
                // var_c = sum(gamma_c * (x - mu)^2, 0) / N_c
                $diffSq = $x->sub($mean)->square();
                $var_c = $diffSq->mulInplace($gamma_c)
                                ->sumAxis(0)
                                ->mulScalarInplace(1.0 / ($N_c_flat[$c] + 1e-8));
                                
                $newVarsList[] = $var_c->expandDims(0); // Shape [1, D]
            }
            // Clip variance to prevent distribution collapse
            $this->vars = Tensor::concat($newVarsList, 0)->clip(1e-6, INF);

            // Update Priors
            for ($c = 0; $c < $this->k; $c++) {
                $this->priors[$c] = $N_c_flat[$c] / $n;
            }

            // Convergence check using Total Log-Likelihood
            $ll = $lse->sum();
            if (abs($ll - $prevLl) < $this->tolerance) break;
            $prevLl = $ll;
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("GMM is not trained.");

        $x = $dataset->samples();
        $logProbsList = [];

        for ($c = 0; $c < $this->k; $c++) {
            $mean = $this->means->row($c);
            $var = $this->vars->row($c);

            $diffSq = $x->sub($mean)->square();
            $term1 = $diffSq->divInplace($var)->sumAxis(1);
            $term2 = $var->mulScalar(2.0 * M_PI)->log()->sum();

            $logProb = $term1->addScalarInplace($term2)->mulScalarInplace(-0.5)->addScalarInplace(log($this->priors[$c]));
            $logProbsList[] = $logProb->expandDims(1);
        }

        $logProbs = Tensor::concat($logProbsList, 1);
        $maxLogProb = $logProbs->maxAxis(1)->expandDims(1);
        $lse = $maxLogProb->add($logProbs->sub($maxLogProb)->exp()->sumAxis(1)->expandDims(1)->log());

        return $logProbs->subInplace($lse)->exp();
    }

    public function predict(Dataset $dataset): Tensor
    {
        // Output the Gaussian Component with the highest probability
        return $this->proba($dataset)->argmax();
    }

    public function trained(): bool
    {
        return $this->means !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['k'=>$this->k,'maxIter'=>$this->maxIter,'tolerance'=>$this->tolerance,'priors'=>$this->priors]));
        $tensors = [];
        if ($this->means !== null) $tensors['means'] = $this->means;
        if ($this->vars  !== null) $tensors['vars']  = $this->vars;
        if ($tensors) SafeTensorsIO::save($dir . '/model.safetensors', $tensors);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int)$c['k'], (int)$c['maxIter'], (float)$c['tolerance']);
        $i->priors = $c['priors'] ?? [];
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->means = $t['means'] ?? null; $i->vars = $t['vars'] ?? null; }
        return $i;
    }
}
