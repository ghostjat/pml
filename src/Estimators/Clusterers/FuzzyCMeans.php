<?php

declare(strict_types=1);

namespace Pml\Estimators\Clusterers;

use Pml\Interfaces\Learner;
use Pml\Lib\SafeTensorsIO;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Probabilistic;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Fuzzy C-Means (Soft Clustering).
 * Assigns points partial membership probabilities to multiple clusters instead of a hard limit.
 * * JIT & Memory Optimized:
 * - Bypasses sequential nested loops by converting the centroid calculations into a 
 * single massive matrix multiplication (U^m_T * X).
 * - Distances and memberships update instantly in native C memory.
 */
final class FuzzyCMeans implements Learner, Probabilistic, Persistable
{
    private int $k;
    private float $fuzziness;
    private int $maxIter;
    private float $tolerance;
    
    private ?Tensor $centroids = null;

    /**
     * @param int $k Number of clusters.
     * @param float $fuzziness Fuzziness parameter (m > 1.0). Higher values mean softer boundaries (default: 2.0).
     */
    public function __construct(int $k = 3, float $fuzziness = 2.0, int $maxIter = 300, float $tolerance = 1e-4)
    {
        if ($fuzziness <= 1.0) throw new \InvalidArgumentException("Fuzziness (m) must be > 1.0.");
        $this->k = $k;
        $this->fuzziness = $fuzziness;
        $this->maxIter = $maxIter;
        $this->tolerance = $tolerance;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = $x->shape()[0];

        // 1. Initialize Memberships (U) randomly and normalize to sum to 1.0 per row
        $U = Tensor::randomUniform([$n, $this->k], 0.1, 1.0);
        $U->divInplace($U->sumAxis(1)->expandDims(1));

        $mTensor = Tensor::zeros(1)->addScalarInplace($this->fuzziness);
        $pTensor = Tensor::zeros(1)->addScalarInplace(-1.0 / ($this->fuzziness - 1.0));

        for ($iter = 0; $iter < $this->maxIter; $iter++) {
            
            // --- UPDATE CENTROIDS ---
            // U_m = U^m
            $U_m = $U->pow($mTensor);
            $U_m_T = $U_m->transpose(); // Shape [K, N]

            // C = (U^m_T * X) / sum(U^m, axis=0)
            // 100% Matrix operation utilizing OpenBLAS GEMM
            $weightedSum = $U_m_T->matmul($x);
            $weights = $U_m_T->sumAxis(1)->expandDims(1);
            $this->centroids = $weightedSum->divInplace($weights);

            // --- UPDATE MEMBERSHIPS ---
            $distList = [];
            for ($c = 0; $c < $this->k; $c++) {
                $cent = $this->centroids->row($c);
                // dist_c = sum((X - c)^2, axis=1) -> Expand to [N, 1]
                $distList[] = $x->sub($cent)->square()->sumAxis(1)->expandDims(1);
            }
            // Combine all distances into [N, K] matrix
            $D = Tensor::concat($distList, 1);

            // W = (D + eps) ^ (-1 / (m - 1))
            $W = $D->addScalar(1e-8)->pow($pTensor);

            // U_new = W / sum(W, axis=1)
            $U_new = $W->divInplace($W->sumAxis(1)->expandDims(1));

            // Check Convergence
            $shift = $U->sub($U_new)->abs()->max();
            $U = $U_new;

            if ($shift < $this->tolerance) break;
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("FuzzyCMeans is not trained.");

        $x = $dataset->samples();
        $pTensor = Tensor::zeros(1)->addScalarInplace(-1.0 / ($this->fuzziness - 1.0));
        
        $distList = [];
        for ($c = 0; $c < $this->k; $c++) {
            $distList[] = $x->sub($this->centroids->row($c))->square()->sumAxis(1)->expandDims(1);
        }
        $D = Tensor::concat($distList, 1);

        $W = $D->addScalar(1e-8)->pow($pTensor);
        return $W->divInplace($W->sumAxis(1)->expandDims(1));
    }

    public function predict(Dataset $dataset): Tensor
    {
        return $this->proba($dataset)->argmaxAxis(1);
    }

    public function trained(): bool
    {
        return $this->centroids !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['k'=>$this->k,'fuzziness'=>$this->fuzziness,'maxIter'=>$this->maxIter,'tolerance'=>$this->tolerance]));
        if ($this->centroids !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['centroids' => $this->centroids]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int)$c['k'], (float)$c['fuzziness'], (int)$c['maxIter'], (float)$c['tolerance']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->centroids = $t['centroids'] ?? null; }
        return $i;
    }
}
