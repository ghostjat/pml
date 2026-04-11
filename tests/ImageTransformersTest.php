<?php

declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Transformers\ImageResizer;
use Pml\Transformers\ImageCropper;
use Pml\Transformers\ColorSpaceConverter;
use InvalidArgumentException;
use RuntimeException;

final class ImageTransformersTest extends TestCase
{
    public function testColorSpaceConverterMathAndShape(): void
    {
        // 1 Batch, 3 Channels (RGB), 2x2 Image
        $r = Tensor::ones(1, 1, 2, 2)->mulScalarInplace(100.0); // Red: 100
        $g = Tensor::ones(1, 1, 2, 2)->mulScalarInplace(50.0);  // Green: 50
        $b = Tensor::ones(1, 1, 2, 2)->mulScalarInplace(10.0);  // Blue: 10
        
        // Concat along the Channel axis (Axis 1) to build the RGB image
        $rgb = Tensor::concat([$r, $g, $b], 1);
        $dataset = new Dataset($rgb);

        $converter = new ColorSpaceConverter();
        $grayDataset = $converter->transform($dataset);
        $grayTensor = $grayDataset->samples();

        // 1. Assert Shape Transition [Batch, 3, H, W] -> [Batch, 1, H, W]
        $this->assertSame([1, 1, 2, 2], $grayTensor->shape());

        // 2. Assert Exact Mathematical Weights
        // Expected = 100 * 0.299 + 50 * 0.587 + 10 * 0.114 = 60.39
        $flat = $grayTensor->toFlatArray();
        foreach ($flat as $pixel) {
            $this->assertEqualsWithDelta(60.39, $pixel, 0.0001);
        }
    }

    public function testColorSpaceConverterRejectsInvalidShapes(): void
    {
        $this->expectException(RuntimeException::class);
        
        // Pass a 2D matrix instead of a 4D image batch
        $invalidSamples = Tensor::ones(10, 10);
        $dataset = new Dataset($invalidSamples);
        
        $converter = new ColorSpaceConverter();
        $converter->transform($dataset);
    }

    public function testImageCropperExtractsExactRegion(): void
    {
        // 1 Batch, 1 Channel, 4x4 Image
        $data = [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16]
                ]
            ]
        ];
        
        $samples = Tensor::fromArray($data);
        $dataset = new Dataset($samples);

        // Crop a 2x2 region starting at X=1, Y=1
        // Expected region:
        // [6, 7]
        // [10, 11]
        $cropper = new ImageCropper(xOffset: 1, yOffset: 1, width: 2, height: 2);
        $cropped = $cropper->transform($dataset)->samples();

        $this->assertSame([1, 1, 2, 2], $cropped->shape());
        
        $flat = $cropped->toFlatArray();
        $this->assertEquals(6.0, $flat[0]);
        $this->assertEquals(7.0, $flat[1]);
        $this->assertEquals(10.0, $flat[2]);
        $this->assertEquals(11.0, $flat[3]);
        
        // Crucial validation: The cropper must return contiguous memory for CNNs
        $this->assertTrue($cropped->isContiguous());
    }

    public function testImageCropperRejectsOutOfBounds(): void
    {
        $this->expectException(InvalidArgumentException::class);
        
        $samples = Tensor::ones(1, 1, 10, 10);
        $dataset = new Dataset($samples);
        
        // Attempt to crop outside the 10x10 bounds
        $cropper = new ImageCropper(xOffset: 8, yOffset: 8, width: 5, height: 5);
        $cropper->transform($dataset);
    }

    public function testImageResizerDownscalesCorrectly(): void
    {
        // 1 Batch, 1 Channel, 4x4 Image
        $data = [
            [
                [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4]
                ]
            ]
        ];
        
        $samples = Tensor::fromArray($data);
        $dataset = new Dataset($samples);

        // Downscale to 2x2
        $resizer = new ImageResizer(targetWidth: 2, targetHeight: 2);
        $resized = $resizer->transform($dataset)->samples();

        $this->assertSame([1, 1, 2, 2], $resized->shape());
        
        // Using Nearest Neighbor, the top-left of each 2x2 block should be selected
        $flat = $resized->toFlatArray();
        $this->assertEquals(1.0, $flat[0]);
        $this->assertEquals(2.0, $flat[1]);
        $this->assertEquals(3.0, $flat[2]);
        $this->assertEquals(4.0, $flat[3]);
    }
}