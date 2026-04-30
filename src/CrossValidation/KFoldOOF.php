<?php
declare(strict_types=1);

namespace Pml\CrossValidation;

use Pml\Dataset;
use Pml\Interfaces\Learner;
use Pml\Tensor;

/**
 * K-Fold cross-validation that produces out-of-fold (OOF) predictions.
 *
 * Trains the model on K-1 folds and predicts the held-out fold each time.
 * Assembles a full OOF prediction tensor [N] in the original row order.
 *
 * @return array{0: Tensor, 1: float[]}  [oof_predictions [N], per-fold log-space RMSE]
 */
final class KFoldOOF
{
    public function __construct(private readonly int $k = 5) {}

    /**
     * @param callable $factory  fn(): Learner — called once per fold for a fresh model
     */
    public function run(callable $factory, Dataset $dataset): array
    {
        $n      = $dataset->numRows();
        $labels = $dataset->labels();
        if ($labels === null) {
            throw new \InvalidArgumentException('KFoldOOF requires a labeled dataset.');
        }

        // OOF buffer: PHP float array, indexed by original row position
        $oofBuffer = array_fill(0, $n, 0.0);

        $foldSize = (int) floor($n / $this->k);
        $foldRmse = [];

        for ($i = 0; $i < $this->k; $i++) {
            $offset = $i * $foldSize;
            $length = ($i === $this->k - 1) ? $n - $offset : $foldSize;

            $val = $dataset->slice($offset, $length);

            // Build train = everything outside this fold
            $trainParts = [];
            if ($offset > 0) {
                $trainParts[] = $dataset->slice(0, $offset);
            }
            $end = $offset + $length;
            if ($end < $n) {
                $trainParts[] = $dataset->slice($end, $n - $end);
            }

            $train = count($trainParts) === 1
                ? $trainParts[0]
                : $trainParts[0]->stack($trainParts[1]);

            $model = $factory();
            $model->train($train);
            $foldPreds = $model->predict($val)->toFlatArray();  // PHP float[]

            // Write into OOF buffer at the correct row positions
            for ($j = 0; $j < $length; $j++) {
                $oofBuffer[$offset + $j] = $foldPreds[$j];
            }

            // Per-fold RMSE in log space (cheap — fold_size rows)
            $foldLabels = $val->labels()->toFlatArray();
            $sse = 0.0;
            for ($j = 0; $j < $length; $j++) {
                $d = $foldPreds[$j] - $foldLabels[$j];
                $sse += $d * $d;
            }
            $foldRmse[] = sqrt($sse / $length);
        }

        return [Tensor::fromArray($oofBuffer), $foldRmse];
    }

    public function k(): int { return $this->k; }
}
