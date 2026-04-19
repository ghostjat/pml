<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Stateful;
use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Ordinal Encoder — maps each specified categorical column's unique values to
 * consecutive integers 0, 1, 2, …
 *
 * All state is stored as Tensors. Transform uses C-level broadcast equal to
 * build the encoded columns without PHP array intermediates.
 *
 * @param int[] $columns Column indices to encode. Empty = encode all columns.
 */
final class OrdinalEncoder implements Transformer, Stateful
{
    /** @var array<int, Tensor> col → sorted unique-values Tensor */
    private array $categories = [];

    public function __construct(private readonly array $columns = []) {}

    public function fit(Dataset $dataset): void
    {
        $X    = $dataset->samples();
        $D    = $X->shape()[1];
        $cols = $this->columns ?: range(0, $D - 1);

        $this->categories = [];
        foreach ($cols as $d) {
            $col = $X->col($d)->copy();  // [N] zero-copy view → own for unique()
            $this->categories[$d] = $col->unique()->sort(0);
        }
    }

    public function transform(Dataset $dataset): Dataset
    {
        if (empty($this->categories)) {
            throw new RuntimeException("OrdinalEncoder has not been fitted.");
        }
        $X    = $dataset->samples();
        $N    = $X->shape()[0];
        $D    = $X->shape()[1];
        $out  = $X->copy();

        foreach ($this->categories as $d => $uniq) {
            $K      = (int)$uniq->size();
            $colIn  = $X->col($d);                                   // [N] zero-copy

            // Encoded[i] = sum_k( k * (colIn[i] == uniq[k]) )
            $encoded = Tensor::zeros($N);
            for ($k = 0; $k < $K; $k++) {
                $val  = Tensor::zeros($N)->addScalarInplace((float)$uniq->buffer()[$k]);
                $mask = $colIn->equal($val);                         // [N] 0/1
                $encoded->addInplace($mask->mulScalarInplace((float)$k));
            }
            // Write encoded column back into output copy
            // Use in-place scatter: multiply out col by 0 then add encoded
            $outCol = $out->col($d);
            // out col is a view — fill via sub/add trick: out_col = out_col - out_col + encoded
            $outCol->subInplace($out->col($d))->addInplace($encoded);
        }

        return new \Pml\Dataset($out, $dataset->labels());
    }

    public function fitted(): bool
    {
        return !empty($this->categories);
    }

    public function getStateDict(string $prefix = ''): array
    {
        $dict = [];
        foreach ($this->categories as $d => $t) {
            $dict["{$prefix}cat.{$d}"] = $t;
        }
        return $dict;
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->categories = [];
        $pfx = "{$prefix}cat.";
        foreach ($dict as $key => $tensor) {
            if (str_starts_with($key, $pfx)) {
                $this->categories[(int)substr($key, \strlen($pfx))] = $tensor;
            }
        }
    }
}
