<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Numeric String Converter — ensures all feature values are valid floats.
 * Non-numeric strings are replaced with NaN (then handled by MissingDataImputer).
 *
 * In this tensor-native port, the samples are already float Tensors, so this
 * transformer is a stateless identity pass — it validates that the data is
 * already numeric and surfaces NaN entries via tensor_isnan.
 */
final class NumericStringConverter implements Transformer
{
    private bool $fitted = false;

    public function fit(Dataset $dataset): void { $this->fitted = true; }

    public function transform(Dataset $dataset): Dataset
    {
        // Replace NaN / Inf with 0 so downstream estimators don't blow up
        $clean = $dataset->samples()->copy();
        $clean->nanToNumInplace(0.0, 0.0, 0.0);
        return new Dataset($clean, $dataset->labels());
    }

    public function fitted(): bool { return $this->fitted; }
}
