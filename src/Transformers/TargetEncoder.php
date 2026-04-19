<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Target Encoder — replaces each categorical value with the mean of the target
 * for that category (supervised encoding).
 *
 * Smoothing: encoded = (count * cat_mean + alpha * global_mean) / (count + alpha)
 * where alpha controls regularization toward the global mean (default 10).
 *
 * All state stored as Tensors. Zero PHP arrays during transform.
 *
 * @param int[]  $columns  Column indices to encode. Empty = encode all columns.
 * @param float  $alpha    Smoothing strength (higher = more regularization).
 */
final class TargetEncoder implements Transformer, Stateful
{
    /** @var array<int, Tensor> col → sorted unique-values Tensor */
    private array $catValues = [];
    /** @var array<int, Tensor> col → target means Tensor (parallel to catValues) */
    private array $catMeans  = [];
    private float $globalMean = 0.0;

    public function __construct(
        private readonly array $columns = [],
        private readonly float $alpha   = 10.0
    ) {}

    public function fit(Dataset $dataset): void
    {
        $X = $dataset->samples();
        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("TargetEncoder requires labeled data.");
        }

        $D    = $X->shape()[1];
        $N    = (float)$X->shape()[0];
        $cols = $this->columns ?: range(0, $D - 1);

        $this->globalMean = $y->sum() / $N;
        $this->catValues  = [];
        $this->catMeans   = [];

        foreach ($cols as $d) {
            $col  = $X->col($d)->copy();
            $uniq = $col->unique()->sort(0);
            $K    = (int)$uniq->size();

            $meansArr = [];
            for ($k = 0; $k < $K; $k++) {
                $catVal  = (float)$uniq->buffer()[$k];
                $valT    = Tensor::zeros((int)$N)->addScalarInplace($catVal);
                $mask    = $col->equal($valT);          // [N] 0/1
                $count   = $mask->sum();
                if ($count < 1.0) {
                    $meansArr[] = $this->globalMean;
                    continue;
                }
                $catMean    = $mask->mul($y)->sum() / $count;
                $smoothed   = ($count * $catMean + $this->alpha * $this->globalMean)
                            / ($count + $this->alpha);
                $meansArr[] = $smoothed;
            }

            $this->catValues[$d] = $uniq;
            $this->catMeans[$d]  = Tensor::fromArray($meansArr);
        }
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (empty($this->catValues)) {
            throw new RuntimeException("TargetEncoder has not been fitted.");
        }
        $X   = $dataset->samples();
        $N   = $X->shape()[0];
        $out = $X->copy();

        foreach ($this->catValues as $d => $uniq) {
            $K       = (int)$uniq->size();
            $colIn   = $X->col($d);
            $means   = $this->catMeans[$d];
            $encoded = Tensor::zeros($N)->addScalarInplace($this->globalMean);

            for ($k = 0; $k < $K; $k++) {
                $catVal  = (float)$uniq->buffer()[$k];
                $mean    = (float)$means->buffer()[$k];
                $valT    = Tensor::zeros($N)->addScalarInplace($catVal);
                $mask    = $colIn->equal($valT);
                // encoded += (mean - globalMean) * mask  (only matched rows change)
                $encoded->addInplace($mask->mulScalarInplace($mean - $this->globalMean));
            }

            $outCol = $out->col($d);
            $outCol->subInplace($out->col($d))->addInplace($encoded);
        }

        return new \Pml\Dataset($out, $dataset->labels());
    }

    public function fitted(): bool
    {
        return !empty($this->catValues);
    }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        foreach ($this->catValues as $d => $t) {
            $dict["{$prefix}catvals.{$d}"] = $t;
        }
        foreach ($this->catMeans as $d => $t) {
            $dict["{$prefix}catmeans.{$d}"] = $t;
        }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->catValues = [];
        $this->catMeans  = [];
        $vPfx = "{$prefix}catvals.";
        $mPfx = "{$prefix}catmeans.";
        foreach ($dict as $key => $tensor) {
            if (str_starts_with($key, $vPfx)) {
                $this->catValues[(int)substr($key, \strlen($vPfx))] = $tensor;
            } elseif (str_starts_with($key, $mPfx)) {
                $this->catMeans[(int)substr($key, \strlen($mPfx))] = $tensor;
            }
        }
    }
}
