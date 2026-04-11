<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Image Vectorizer — flattens multi-channel image tensors to 1-D feature vectors
 * and normalizes pixel values to [0, 1].
 *
 * JIT & Memory Optimized:
 * - Flatten is a zero-copy C reshape (tensor_reshape with -1).
 * - Pixel normalization is a single C scalar multiply.
 */
final class ImageVectorizer implements Transformer
{
    private bool $fitted = false;

    /** @param int $maxPixelValue  Typical: 255 for uint8 images */
    public function __construct(private readonly float $maxPixelValue = 255.0) {}

    public function fit(Dataset $dataset): void { $this->fitted = true; }

    public function transform(Dataset $dataset): Dataset
    {
        $x = $dataset->samples();

        // If already 2-D [N × features], just normalize
        if ($x->ndim() === 2) {
            return new Dataset(
                $x->mulScalar(1.0 / $this->maxPixelValue),
                $dataset->labels()
            );
        }

        // 3-D [N × H × W] or 4-D [N × C × H × W] — flatten each sample
        $n    = $x->shape()[0];
        $flat = $x->reshape($n, (int)($x->size() / $n));          // [N × H*W*C]
        return new Dataset(
            $flat->mulScalar(1.0 / $this->maxPixelValue),
            $dataset->labels()
        );
    }

    public function fitted(): bool { return $this->fitted; }
}
