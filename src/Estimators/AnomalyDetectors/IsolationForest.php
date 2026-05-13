<?php

declare(strict_types=1);

namespace Pml\Estimators\AnomalyDetectors;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Isolation Forest Anomaly Detector.
 * Identifies anomalies by isolating observations randomly. Anomalies require shorter paths to isolate.
 * * JIT & Memory Optimized:
 * - Employs purely random AVX2-accelerated thresholds.
 * - Extracts split masks locally to route data instantly without FFI bottlenecks.
 * - Caches anomaly scores for zero-copy decision thresholding.
 */
final class IsolationForest implements Learner, Persistable
{
    private int $nEstimators;
    private int $sampleSize;
    private float $contamination;

    private array $trees = [];
    private float $threshold  = 0.5;
    private int   $nFeatures  = 0;
    private float $cNorm      = 1.0;

    // Flat C-resident tensors for batch scoring (built after train())
    private ?Tensor $flatFeats  = null;
    private ?Tensor $flatThresh = null;
    private ?Tensor $flatLefts  = null;
    private ?Tensor $flatRights = null;
    private ?Tensor $flatLsizes = null;
    private ?Tensor $treeSizes  = null;

    /**
     * @param int $nEstimators Number of isolation trees to build.
     * @param int $sampleSize Number of samples drawn to build each tree (standard is 256).
     * @param float $contamination Expected proportion of anomalies in the dataset (e.g., 0.1 for 10%).
     */
    public function __construct(int $nEstimators = 100, int $sampleSize = 256, float $contamination = 0.1)
    {
        $this->nEstimators = $nEstimators;
        $this->sampleSize = $sampleSize;
        $this->contamination = $contamination;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = $x->shape()[0];
        $this->nFeatures = $x->shape()[1];

        // The maximum depth of an isolation tree is typically ceiling(log2(sampleSize))
        $maxDepth = (int) ceil(log($this->sampleSize, 2));

        for ($i = 0; $i < $this->nEstimators; $i++) {
            $curSize = min($this->sampleSize, $n);
            $idxT = Tensor::range(0, $n - 1, 1)->randomPermutation()->slice(0, 0, $curSize);
            $subX = $x->take($idxT, 0);
            $this->trees[] = $this->buildTree($subX, 0, $maxDepth);
        }

        // Serialize PHP trees → flat C tensors for all-C batch scoring
        $this->serializeTrees();

        // Calibrate threshold from training-set scores
        $this->calibrateThreshold($dataset);
    }

    /**
     * Recursively partitions the data using random C-Level axis splits.
     */
    private function buildTree(Tensor $x, int $currentDepth, int $maxDepth): array
    {
        $n = $x->shape()[0];

        // Terminal Node (Max Depth or completely isolated)
        if ($currentDepth >= $maxDepth || $n <= 1) {
            return ['size' => $n];
        }

        $f = random_int(0, $this->nFeatures - 1);
        $col = $x->col($f);
        
        $min = $col->min();
        $max = $col->max();

        // If the chosen feature is constant, attempt to find a non-constant one
        $attempts = 0;
        while ($min === $max && $attempts < $this->nFeatures) {
            $f = ($f + 1) % $this->nFeatures;
            $col = $x->col($f);
            $min = $col->min();
            $max = $col->max();
            $attempts++;
        }

        // Entire subspace is uniform, stop building
        if ($min === $max) {
            return ['size' => $n];
        }

        // Select a completely random split threshold
        $split = $min + (mt_rand() / mt_getrandmax() * ($max - $min));

        // Generate SIMD boolean mask for the split
        $threshT = Tensor::zeros($n)->addScalarInplace($split);
        $leftMask = $col->less($threshT);

        // Extract mask to PHP to route rows instantly
        $maskArray = $leftMask->toFlatArray();
        $leftIdx = [];
        $rightIdx = [];
        
        foreach ($maskArray as $i => $val) {
            if ($val > 0.5) $leftIdx[] = $i;
            else $rightIdx[] = $i;
        }

        // If a random split somehow failed to divide the data (e.g., float precision), force a leaf
        if (empty($leftIdx) || empty($rightIdx)) {
            return ['size' => $n];
        }

        $leftT = Tensor::fromArray($leftIdx);
        $rightT = Tensor::fromArray($rightIdx);

        return [
            'feature'   => $f,
            'threshold' => $split,
            'left'      => $this->buildTree($x->take($leftT, 0), $currentDepth + 1, $maxDepth),
            'right'     => $this->buildTree($x->take($rightT, 0), $currentDepth + 1, $maxDepth),
        ];
    }

    /**
     * Uses the trained trees to predict Anomalies.
     * Returns 1.0 for Anomaly, and 0.0 for Normal.
     */
    public function predict(Dataset $dataset): Tensor
    {
        // threshold in C via greaterScalar — zero PHP per-sample work
        return $this->anomalyScores($dataset)->greaterScalar($this->threshold);
    }

    /**
     * Computes the raw anomaly score in [0,1] for each sample.
     * Delegates entirely to C (OpenMP parallel over N, zero PHP per-sample work).
     */
    public function anomalyScores(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Isolation Forest is not trained.");
        }
        if ($this->flatFeats === null) {
            $this->serializeTrees();
        }
        return Tensor::iforestScore(
            $dataset->samples(),
            $this->flatFeats, $this->flatThresh,
            $this->flatLefts, $this->flatRights,
            $this->flatLsizes, $this->treeSizes,
            $this->cNorm
        );
    }

    /**
     * Serialize the PHP recursive tree array to flat C tensors.
     * BFS traversal; runs once after training.
     */
    private function serializeTrees(): void
    {
        $T = $this->nEstimators;
        if ($T === 0) return;

        // Compute max possible nodes per tree: 2^(maxDepth+1)
        $maxD     = (int)ceil(log($this->sampleSize, 2));
        $maxNodes = (1 << ($maxD + 1)) + 2;   // conservative upper bound

        $totalNodes = $T * $maxNodes;
        $feats  = array_fill(0, $totalNodes, -1.0);
        $thresh = array_fill(0, $totalNodes,  0.0);
        $lefts  = array_fill(0, $totalNodes, -1.0);
        $rights = array_fill(0, $totalNodes, -1.0);
        $lsizes = array_fill(0, $totalNodes,  1.0);
        $tsizes = array_fill(0, $T, 0.0);

        foreach ($this->trees as $t => $tree) {
            $offset = $t * $maxNodes;
            $queue  = [[$tree, -1, false]];
            $head   = 0;
            $nextId = 0;
            while ($head < count($queue)) {
                [$node, $parentId, $isLeft] = $queue[$head++];
                $myId = $nextId++;
                if ($parentId >= 0) {
                    if ($isLeft) $lefts[$offset + $parentId]  = (float)$myId;
                    else         $rights[$offset + $parentId] = (float)$myId;
                }
                if (isset($node['feature'])) {
                    $feats[$offset + $myId]  = (float)$node['feature'];
                    $thresh[$offset + $myId] = (float)$node['threshold'];
                    $queue[] = [$node['left'],  $myId, true];
                    $queue[] = [$node['right'], $myId, false];
                } else {
                    // Leaf: feature=-1 sentinel, store leaf size for c() adjustment
                    $lsizes[$offset + $myId] = (float)($node['size'] ?? 1);
                }
            }
            $tsizes[$t] = (float)$nextId;
        }

        $this->cNorm      = $this->c($this->sampleSize);
        $this->flatFeats  = Tensor::fromArray($feats);
        $this->flatThresh = Tensor::fromArray($thresh);
        $this->flatLefts  = Tensor::fromArray($lefts);
        $this->flatRights = Tensor::fromArray($rights);
        $this->flatLsizes = Tensor::fromArray($lsizes);
        $this->treeSizes  = Tensor::fromArray($tsizes);
    }

    /**
     * Evaluates the training data to define the dynamic threshold boundary based on $contamination.
     */
    private function calibrateThreshold(Dataset $dataset): void
    {
        $scores = $this->anomalyScores($dataset)->toFlatArray();
        rsort($scores); // Sort descending (Highest anomalies first)

        // Find the boundary score that isolates the top X% of the data
        $thresholdIndex = (int) floor(count($scores) * $this->contamination);
        
        if ($thresholdIndex >= count($scores)) {
            $thresholdIndex = count($scores) - 1;
        }

        $this->threshold = $scores[$thresholdIndex];
    }

    /**
     * The average path length of an unsuccessful search in a Binary Search Tree (BST).
     */
    private function c(int $n): float
    {
        if ($n <= 1) return 0.0;
        if ($n === 2) return 1.0;
        // 2 * (ln(n-1) + Euler's constant) - (2*(n-1)/n)
        return 2.0 * (log($n - 1) + 0.5772156649) - (2.0 * ($n - 1) / $n);
    }

    public function trained(): bool
    {
        return !empty($this->trees);
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['nEstimators' => $this->nEstimators, 'sampleSize' => $this->sampleSize, 'contamination' => $this->contamination, 'threshold' => $this->threshold, 'nFeatures' => $this->nFeatures]));
        file_put_contents($dir . '/trees.json', json_encode($this->trees));
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int) $c['nEstimators'], (int) $c['sampleSize'], (float) $c['contamination']);
        $i->threshold = (float) $c['threshold'];
        $i->nFeatures = (int) $c['nFeatures'];
        $i->trees = json_decode(file_get_contents($dir . '/trees.json'), true);
        $i->serializeTrees();
        return $i;
    }
}