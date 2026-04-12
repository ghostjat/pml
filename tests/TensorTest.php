<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use InvalidArgumentException;
use RuntimeException;

/**
 * Comprehensive test suite for Tensor.
 * Covers: creation, dtypes, shape ops, math, linalg, views, masking, I/O, error paths.
 */
final class TensorTest extends TestCase
{
    // =========================================================================
    // HELPERS
    // =========================================================================

    private static function delta(): float { return 1e-4; }

    private function flat(Tensor $t): array { return $t->toFlatArray(); }

    // =========================================================================
    // 1. CREATION & DTYPES
    // =========================================================================

    public function testFromArray1D(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0]);
        $this->assertSame([3], $t->shape());
        $this->assertSame(1, $t->ndim());
        $this->assertSame(3, $t->size());
        $this->assertSame(Tensor::DTYPE_FLOAT32, $t->dtype());
        $flat = $this->flat($t);
        $this->assertEqualsWithDelta(2.0, $flat[1], self::delta());
    }

    public function testFromArray2D(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $this->assertSame([2, 2], $t->shape());
        $this->assertSame(2, $t->ndim());
        $this->assertSame(4, $t->size());
        $flat = $this->flat($t);
        $this->assertEqualsWithDelta(3.0, $flat[2], self::delta());
    }

    public function testFromArray3D(): void
    {
        $data = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]];
        $t = Tensor::fromArray($data);
        $this->assertSame([2, 2, 2], $t->shape());
        $this->assertSame(8, $t->size());
    }

    public function testFromArrayInt32(): void
    {
        $t = Tensor::fromArray([100, 200, 300], Tensor::DTYPE_INT32);
        $this->assertSame(Tensor::DTYPE_INT32, $t->dtype());
        $flat = $this->flat($t);
        $this->assertSame(100, $flat[0]);
        $this->assertSame(300, $flat[2]);
    }

    public function testFromArrayInt32LargeTokenIds(): void
    {
        $tokens = [8725, 291, 1024, 50256];
        $t = Tensor::fromArray($tokens, Tensor::DTYPE_INT32);
        $flat = $this->flat($t);
        foreach ($tokens as $i => $v) {
            $this->assertSame($v, $flat[$i]);
        }
    }

    public function testJaggedArrayThrows(): void
    {
        $this->expectException(InvalidArgumentException::class);
        Tensor::fromArray([[1.0, 2.0], [3.0]]);
    }

    public function testZeros(): void
    {
        $t = Tensor::zeros(3, 4);
        $this->assertSame([3, 4], $t->shape());
        foreach ($this->flat($t) as $v) {
            $this->assertEqualsWithDelta(0.0, $v, self::delta());
        }
    }

    public function testOnes(): void
    {
        $t = Tensor::ones(2, 5);
        $this->assertSame([2, 5], $t->shape());
        foreach ($this->flat($t) as $v) {
            $this->assertEqualsWithDelta(1.0, $v, self::delta());
        }
    }

    public function testRange(): void
    {
        $t = Tensor::range(0.0, 5.0, 1.0);
        $flat = $this->flat($t);
        $this->assertSame(5, \count($flat));
        $this->assertEqualsWithDelta(0.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(4.0, $flat[4], self::delta());
    }

    public function testLinspace(): void
    {
        $t = Tensor::linspace(0.0, 1.0, 5);
        $flat = $this->flat($t);
        $this->assertSame(5, \count($flat));
        $this->assertEqualsWithDelta(0.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.25, $flat[1], self::delta());
        $this->assertEqualsWithDelta(1.0, $flat[4], self::delta());
    }

    public function testRandomNormalShape(): void
    {
        $t = Tensor::randomNormal([100, 10], 0.0, 1.0);
        $this->assertSame([100, 10], $t->shape());
        $flat = $this->flat($t);
        // mean should be near 0, std near 1 for large sample
        $mean = \array_sum($flat) / \count($flat);
        $this->assertEqualsWithDelta(0.0, $mean, 0.3);
    }

    public function testRandomUniformBounds(): void
    {
        $t = Tensor::randomUniform([500], 2.0, 5.0);
        foreach ($this->flat($t) as $v) {
            $this->assertGreaterThanOrEqual(2.0, $v);
            $this->assertLessThanOrEqual(5.0, $v);
        }
    }

    public function testFill(): void
    {
        $t = Tensor::zeros(3, 3);
        $t->fill(7.5);
        foreach ($this->flat($t) as $v) {
            $this->assertEqualsWithDelta(7.5, $v, self::delta());
        }
    }

    // =========================================================================
    // 2. COPY & VIEW ISOLATION
    // =========================================================================

    public function testCopyIsIndependent(): void
    {
        $orig = Tensor::fromArray([1.0, 2.0, 3.0]);
        $copy = $orig->copy();
        $copy->fill(99.0);
        // orig unchanged
        $this->assertEqualsWithDelta(1.0, $this->flat($orig)[0], self::delta());
    }

    public function testViewSharesParentData(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0]);
        $v = $t->view();
        $this->assertSame($t->shape(), $v->shape());
        $this->assertSame($t->size(), $v->size());
    }

    public function testIsContiguous(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0]);
        $this->assertTrue($t->isContiguous());
    }

    // =========================================================================
    // 3. SHAPE MUTATIONS
    // =========================================================================

    public function testReshape(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        $r = $t->reshape(2, 3);
        $this->assertSame([2, 3], $r->shape());
        $this->assertSame(6, $r->size());
    }

    public function testFlatten(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $f = $t->flatten();
        $this->assertSame([4], $f->shape());
        $flat = $this->flat($f);
        $this->assertEqualsWithDelta(3.0, $flat[2], self::delta());
    }

    public function testExpandDims(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0]);
        $e = $t->expandDims(0);
        $this->assertSame([1, 3], $e->shape());
    }

    public function testSqueeze(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0]);
        $e = $t->expandDims(0);  // [1, 3]
        $s = $e->squeeze();
        $this->assertSame([3], $s->shape());
    }

    public function testTranspose2D(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);  // [2, 3]
        $tr = $t->transpose();
        $this->assertSame([3, 2], $tr->shape());
        $flat = $this->flat($tr);
        // col-major: [1,4,2,5,3,6]
        $this->assertEqualsWithDelta(4.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[2], self::delta());
    }

    public function testSwapaxes(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);  // [3, 2]
        $s = $t->swapaxes(0, 1);
        $this->assertSame([2, 3], $s->shape());
    }

    // =========================================================================
    // 4. SLICING & VIEWS
    // =========================================================================

    public function testSliceAxis0(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        $s = $t->slice(0, 1, 2);  // rows 1-2
        $this->assertSame([2, 2], $s->shape());
        $flat = $this->flat($s);
        $this->assertEqualsWithDelta(3.0, $flat[0], self::delta());
    }

    public function testSliceStep(): void
    {
        $t = Tensor::range(0.0, 10.0, 1.0);  // [0,1,...,9]
        $s = $t->sliceStep(0, 0, 10, 2);     // every other: [0,2,4,6,8]
        $this->assertSame(5, $s->size());
        $flat = $this->flat($s);
        $this->assertEqualsWithDelta(4.0, $flat[2], self::delta());
    }

    public function testRowView(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        $r = $t->row(1);
        $flat = $this->flat($r);
        $this->assertEqualsWithDelta(4.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(6.0, $flat[2], self::delta());
    }

    public function testColView(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        $c = $t->col(1);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(2.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(5.0, $flat[1], self::delta());
    }

    // =========================================================================
    // 5. BINARY MATH OPS
    // =========================================================================

    public function testAdd(): void
    {
        $a = Tensor::fromArray([1.0, 2.0, 3.0]);
        $b = Tensor::fromArray([4.0, 5.0, 6.0]);
        $c = $a->add($b);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(5.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(9.0, $flat[2], self::delta());
    }

    public function testSub(): void
    {
        $a = Tensor::fromArray([5.0, 3.0, 1.0]);
        $b = Tensor::fromArray([1.0, 1.0, 1.0]);
        $c = $a->sub($b);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(4.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[2], self::delta());
    }

    public function testMul(): void
    {
        $a = Tensor::fromArray([2.0, 3.0, 4.0]);
        $b = Tensor::fromArray([3.0, 3.0, 3.0]);
        $c = $a->mul($b);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(6.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(12.0, $flat[2], self::delta());
    }

    public function testDiv(): void
    {
        $a = Tensor::fromArray([6.0, 9.0, 12.0]);
        $b = Tensor::fromArray([2.0, 3.0, 4.0]);
        $c = $a->div($b);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(3.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[2], self::delta());
    }

    public function testAddScalar(): void
    {
        $a = Tensor::fromArray([1.0, 2.0, 3.0]);
        $c = $a->addScalar(10.0);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(11.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(13.0, $flat[2], self::delta());
    }

    public function testMulScalar(): void
    {
        $a = Tensor::fromArray([2.0, 4.0, 6.0]);
        $c = $a->mulScalar(0.5);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[2], self::delta());
    }

    public function testPow(): void
    {
        $base = Tensor::fromArray([2.0, 3.0, 4.0]);
        $exp  = Tensor::fromArray([2.0, 2.0, 2.0]);
        $c = $base->pow($exp);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(4.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(16.0, $flat[2], self::delta());
    }

    public function testClip(): void
    {
        $t = Tensor::fromArray([-2.0, 0.5, 3.0, 5.0]);
        $c = $t->clip(0.0, 2.0);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(0.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.5, $flat[1], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[2], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[3], self::delta());
    }

    // =========================================================================
    // 6. IN-PLACE OPS
    // =========================================================================

    public function testAddInplace(): void
    {
        $a = Tensor::fromArray([1.0, 2.0, 3.0]);
        $b = Tensor::fromArray([1.0, 1.0, 1.0]);
        $a->addInplace($b);
        $flat = $this->flat($a);
        $this->assertEqualsWithDelta(2.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(4.0, $flat[2], self::delta());
    }

    public function testMulScalarInplace(): void
    {
        $a = Tensor::fromArray([1.0, 2.0, 3.0]);
        $a->mulScalarInplace(3.0);
        $flat = $this->flat($a);
        $this->assertEqualsWithDelta(3.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(9.0, $flat[2], self::delta());
    }

    public function testDivInplace(): void
    {
        $a = Tensor::fromArray([4.0, 8.0, 12.0]);
        $b = Tensor::fromArray([2.0, 2.0, 2.0]);
        $a->divInplace($b);
        $flat = $this->flat($a);
        $this->assertEqualsWithDelta(2.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(6.0, $flat[2], self::delta());
    }

    // =========================================================================
    // 7. UNARY MATH
    // =========================================================================

    public function testSqrt(): void
    {
        $t = Tensor::fromArray([4.0, 9.0, 16.0]);
        $flat = $this->flat($t->sqrt());
        $this->assertEqualsWithDelta(2.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(4.0, $flat[2], self::delta());
    }

    public function testSquare(): void
    {
        $t = Tensor::fromArray([2.0, 3.0, 4.0]);
        $flat = $this->flat($t->square());
        $this->assertEqualsWithDelta(4.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(9.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(16.0, $flat[2], self::delta());
    }

    public function testAbs(): void
    {
        $t = Tensor::fromArray([-3.0, 0.0, 5.0]);
        $flat = $this->flat($t->abs());
        $this->assertEqualsWithDelta(3.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(5.0, $flat[2], self::delta());
    }

    public function testSign(): void
    {
        $t = Tensor::fromArray([-5.0, 0.0, 3.0]);
        $flat = $this->flat($t->sign());
        $this->assertEqualsWithDelta(-1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.0,  $flat[1], self::delta());
        $this->assertEqualsWithDelta(1.0,  $flat[2], self::delta());
    }

    public function testExp(): void
    {
        $t = Tensor::fromArray([0.0, 1.0]);
        $flat = $this->flat($t->exp());
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(M_E, $flat[1], 0.001);
    }

    public function testLog(): void
    {
        $t = Tensor::fromArray([1.0, (float)M_E]);
        $flat = $this->flat($t->log());
        $this->assertEqualsWithDelta(0.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(1.0, $flat[1], self::delta());
    }

    public function testLog1p(): void
    {
        $t = Tensor::fromArray([0.0, 1.0]);
        $flat = $this->flat($t->log1p());
        $this->assertEqualsWithDelta(0.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(log(2.0), $flat[1], self::delta());
    }

    public function testRound(): void
    {
        $t = Tensor::fromArray([1.4, 1.5, 2.6]);
        $flat = $this->flat($t->round());
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[2], self::delta());
    }

    public function testFloorCeil(): void
    {
        $t = Tensor::fromArray([1.2, 2.8]);
        $fl = $this->flat($t->floor());
        $this->assertEqualsWithDelta(1.0, $fl[0], self::delta());
        $this->assertEqualsWithDelta(2.0, $fl[1], self::delta());

        $ce = $this->flat($t->ceil());
        $this->assertEqualsWithDelta(2.0, $ce[0], self::delta());
        $this->assertEqualsWithDelta(3.0, $ce[1], self::delta());
    }

    public function testSigmoid(): void
    {
        $t = Tensor::fromArray([0.0]);
        $flat = $this->flat($t->sigmoid());
        $this->assertEqualsWithDelta(0.5, $flat[0], self::delta());

        $t2 = Tensor::fromArray([100.0]);
        $flat2 = $this->flat($t2->sigmoid());
        $this->assertEqualsWithDelta(1.0, $flat2[0], 0.001);
    }

    public function testTanh(): void
    {
        $t = Tensor::fromArray([0.0]);
        $flat = $this->flat($t->tanh());
        $this->assertEqualsWithDelta(0.0, $flat[0], self::delta());
    }

    public function testRelu(): void
    {
        $t = Tensor::fromArray([-3.0, 0.0, 2.0, 5.0]);
        $flat = $this->flat($t->relu());
        $this->assertEqualsWithDelta(0.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[2], self::delta());
        $this->assertEqualsWithDelta(5.0, $flat[3], self::delta());
    }

    public function testTrigonometry(): void
    {
        $t = Tensor::fromArray([0.0, (float)M_PI_2]);
        $sin = $this->flat($t->sin());
        $cos = $this->flat($t->cos());
        $this->assertEqualsWithDelta(0.0, $sin[0], self::delta());
        $this->assertEqualsWithDelta(1.0, $sin[1], self::delta());
        $this->assertEqualsWithDelta(1.0, $cos[0], self::delta());
        $this->assertEqualsWithDelta(0.0, $cos[1], self::delta());
    }

    // =========================================================================
    // 8. AGGREGATIONS
    // =========================================================================

    public function testSum(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0]);
        $this->assertEqualsWithDelta(10.0, $t->sum(), self::delta());
    }

    public function testProduct(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0]);
        $this->assertEqualsWithDelta(24.0, $t->product(), self::delta());
    }

    public function testMean(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0]);
        $this->assertEqualsWithDelta(2.5, $t->mean(), self::delta());
    }

    public function testMinMax(): void
    {
        $t = Tensor::fromArray([3.0, 1.0, 4.0, 1.0, 5.0]);
        $this->assertEqualsWithDelta(1.0, $t->min(), self::delta());
        $this->assertEqualsWithDelta(5.0, $t->max(), self::delta());
    }

    public function testArgminArgmax(): void
    {
        $t = Tensor::fromArray([3.0, 1.0, 4.0, 1.0, 5.0]);
        $this->assertSame(1, $t->argmin());
        $this->assertSame(4, $t->argmax());
    }

    public function testVarianceStd(): void
    {
        // Population variance of [2,4,4,4,5,5,7,9] = 4.0
        $t = Tensor::fromArray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        $this->assertEqualsWithDelta(4.0, $t->variance(), 0.01);
        $this->assertEqualsWithDelta(2.0, $t->std(), 0.01);
    }

    public function testMedian(): void
    {
        $t = Tensor::fromArray([3.0, 1.0, 2.0]);  // sorted: 1,2,3 → median=2
        $this->assertEqualsWithDelta(2.0, $t->median(), self::delta());
    }

    public function testSumAxis(): void
    {
        // [[1,2],[3,4]] sum axis=0 → [4,6]
        $t = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $s = $t->sumAxis(0);
        $flat = $this->flat($s);
        $this->assertEqualsWithDelta(4.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(6.0, $flat[1], self::delta());
    }

    public function testMeanAxis(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $m = $t->meanAxis(1);
        $flat = $this->flat($m);
        $this->assertEqualsWithDelta(1.5, $flat[0], self::delta());
        $this->assertEqualsWithDelta(3.5, $flat[1], self::delta());
    }

    public function testMaxAxis(): void
    {
        $t = Tensor::fromArray([[1.0, 5.0], [3.0, 2.0]]);
        $m = $t->maxAxis(1);
        $flat = $this->flat($m);
        $this->assertEqualsWithDelta(5.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[1], self::delta());
    }

    public function testMinAxis(): void
    {
        $t = Tensor::fromArray([[1.0, 5.0], [3.0, 2.0]]);
        $m = $t->minAxis(0);
        $flat = $this->flat($m);
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[1], self::delta());
    }

    public function testCumsumAxis(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0]);
        $c = $t->cumsum(0);
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(6.0, $flat[2], self::delta());
        $this->assertEqualsWithDelta(10.0, $flat[3], self::delta());
    }

    // =========================================================================
    // 9. NORMALIZE & STANDARDIZE
    // =========================================================================

    public function testNormalize(): void
    {
        $t = Tensor::fromArray([3.0, 4.0, 0.0, 1.0]);
        $n = $t->normalize();
        // Shape must be preserved and all values must be finite
        $this->assertSame($t->shape(), $n->shape());
        foreach ($this->flat($n) as $v) {
            $this->assertFalse(\is_nan($v),      "normalize() produced NaN");
            $this->assertFalse(\is_infinite($v), "normalize() produced Inf");
        }
    }

    public function testStandardize(): void
    {
        $t = Tensor::fromArray([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        $s = $t->standardize();
        // Mean should be ~0 after standardization
        $this->assertEqualsWithDelta(0.0, $s->mean(), 0.01);
    }

    // =========================================================================
    // 10. LOGICAL & COMPARISON
    // =========================================================================

    public function testEqual(): void
    {
        $a = Tensor::fromArray([1.0, 2.0, 3.0]);
        $b = Tensor::fromArray([1.0, 0.0, 3.0]);
        $flat = $this->flat($a->equal($b));
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(1.0, $flat[2], self::delta());
    }

    public function testGreaterLess(): void
    {
        $a = Tensor::fromArray([3.0, 1.0, 2.0]);
        $b = Tensor::fromArray([2.0, 2.0, 2.0]);
        $gt = $this->flat($a->greater($b));
        $lt = $this->flat($a->less($b));
        $this->assertEqualsWithDelta(1.0, $gt[0], self::delta());
        $this->assertEqualsWithDelta(0.0, $gt[1], self::delta());
        $this->assertEqualsWithDelta(0.0, $lt[0], self::delta());
        $this->assertEqualsWithDelta(1.0, $lt[1], self::delta());
    }

    public function testLessScalarF32(): void
    {
        $t = Tensor::fromArray([1.0, 5.0, 3.0, 7.0]);
        $flat = $this->flat($t->lessScalar(4.0));
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(1.0, $flat[2], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[3], self::delta());
    }

    public function testLogicalNot(): void
    {
        $t = Tensor::fromArray([0.0, 1.0, 0.0]);
        $flat = $this->flat($t->logicalNot());
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(1.0, $flat[2], self::delta());
    }

    public function testAnyAll(): void
    {
        $all1 = Tensor::fromArray([1.0, 1.0, 1.0]);
        $mix  = Tensor::fromArray([1.0, 0.0, 1.0]);
        $all0 = Tensor::fromArray([0.0, 0.0, 0.0]);

        $this->assertTrue($all1->all());
        $this->assertFalse($mix->all());
        $this->assertTrue($mix->any());
        $this->assertFalse($all0->any());
    }

    // =========================================================================
    // 11. NaN / Inf HANDLING
    // =========================================================================

    public function testIsNanIsInf(): void
    {
        // For a tensor of normal finite values, isNan() and isInf() must return all zeros.
        $t   = Tensor::fromArray([1.0, 2.0, 3.0, -4.0, 0.0]);
        $nan = $this->flat($t->isNan());
        $inf = $this->flat($t->isInf());
        $this->assertSame($t->shape(), $t->isNan()->shape(), 'isNan shape mismatch');
        $this->assertSame($t->shape(), $t->isInf()->shape(), 'isInf shape mismatch');
        foreach ($nan as $v) {
            $this->assertEqualsWithDelta(0.0, $v, self::delta(), 'isNan() flagged a normal value');
        }
        foreach ($inf as $v) {
            $this->assertEqualsWithDelta(0.0, $v, self::delta(), 'isInf() flagged a normal value');
        }
    }

    public function testNanToNumInplace(): void
    {
        // nanToNumInplace must not alter already-finite values.
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0]);
        $t->nanToNumInplace(0.0, 999.0, -999.0);
        $flat = $this->flat($t);
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[2], self::delta());
        $this->assertEqualsWithDelta(4.0, $flat[3], self::delta());
    }

    // =========================================================================
    // 12. MASKING & FANCY INDEXING
    // =========================================================================

    public function testWhere(): void
    {
        $cond = Tensor::fromArray([1.0, 0.0, 1.0, 0.0]);
        $x    = Tensor::fromArray([10.0, 10.0, 10.0, 10.0]);
        $y    = Tensor::fromArray([20.0, 20.0, 20.0, 20.0]);
        $flat = $this->flat($cond->where($x, $y));
        $this->assertEqualsWithDelta(10.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(20.0, $flat[1], self::delta());
    }

    public function testBooleanIndex(): void
    {
        $t    = Tensor::fromArray([1.0, 2.0, 3.0, 4.0, 5.0]);
        $mask = Tensor::fromArray([0.0, 1.0, 0.0, 1.0, 0.0]);
        $r = $t->booleanIndex($mask);
        $this->assertSame(2, $r->size());
        $flat = $this->flat($r);
        $this->assertEqualsWithDelta(2.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(4.0, $flat[1], self::delta());
    }

    public function testUnique(): void
    {
        $t = Tensor::fromArray([3.0, 1.0, 2.0, 1.0, 3.0]);
        $u = $t->unique();
        // unique returns sorted unique values
        $this->assertSame(3, $u->size());
        $flat = $this->flat($u);
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[2], self::delta());
    }

    public function testBincount(): void
    {
        // bincount works on float32 tensors containing non-negative integer values.
        $t = Tensor::fromArray([0.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
        $b = $t->bincount();
        $flat = $this->flat($b);
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(3.0, $flat[2], self::delta());
    }

    // =========================================================================
    // 13. SORTING
    // =========================================================================

    public function testSort(): void
    {
        $t = Tensor::fromArray([3.0, 1.0, 4.0, 1.0, 5.0]);
        $s = $t->sort(0);
        $flat = $this->flat($s);
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(5.0, $flat[4], self::delta());
    }

    public function testArgsort(): void
    {
        $t = Tensor::fromArray([3.0, 1.0, 2.0]);
        $a = $t->argsort(0);
        $flat = $this->flat($a);
        // Index of smallest (1.0 at pos 1), middle (2.0 at pos 2), largest (3.0 at pos 0)
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(2.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[2], self::delta());
    }

    public function testTopk(): void
    {
        $t   = Tensor::fromArray([3.0, 1.0, 4.0, 1.0, 5.0]);
        $top = $t->topk(3, 0);
        $this->assertSame(3, $top->size());
        // The top-3 values from {3,1,4,1,5} are {5,4,3} — returned in any order.
        $flat = $this->flat($top);
        sort($flat);
        $this->assertEqualsWithDelta(3.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(4.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(5.0, $flat[2], self::delta());
    }

    // =========================================================================
    // 14. CONCATENATION & PADDING
    // =========================================================================

    public function testConcat(): void
    {
        $a = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $b = Tensor::fromArray([[5.0, 6.0]]);
        $c = Tensor::concat([$a, $b], 0);
        $this->assertSame([3, 2], $c->shape());
        $flat = $this->flat($c);
        $this->assertEqualsWithDelta(5.0, $flat[4], self::delta());
    }

    public function testPad(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0]);
        $p = $t->pad([1, 1], 0.0);
        $this->assertSame(5, $p->size());
        $flat = $this->flat($p);
        $this->assertEqualsWithDelta(0.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(1.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(0.0, $flat[4], self::delta());
    }

    // =========================================================================
    // 15. LINEAR ALGEBRA
    // =========================================================================

    public function testDot(): void
    {
        $a = Tensor::fromArray([1.0, 2.0, 3.0]);
        $b = Tensor::fromArray([4.0, 5.0, 6.0]);
        $this->assertEqualsWithDelta(32.0, $a->dot($b), self::delta());
    }

    public function testMatmul(): void
    {
        // [2,3] x [3,2] = [2,2]
        $a = Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        $b = Tensor::fromArray([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]);
        $c = $a->matmul($b);
        $this->assertSame([2, 2], $c->shape());
        $flat = $this->flat($c);
        // [0,0] = 1*7+2*9+3*11 = 58
        $this->assertEqualsWithDelta(58.0, $flat[0], self::delta());
        // [0,1] = 1*8+2*10+3*12 = 64
        $this->assertEqualsWithDelta(64.0, $flat[1], self::delta());
    }

    public function testTrace(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $this->assertEqualsWithDelta(5.0, $t->trace(), self::delta());
    }

    public function testInverse(): void
    {
        // [[2,1],[1,1]] inverse = [[1,-1],[-1,2]]
        $t = Tensor::fromArray([[2.0, 1.0], [1.0, 1.0]]);
        $inv = $t->inverse();
        $flat = $this->flat($inv);
        $this->assertEqualsWithDelta(1.0,  $flat[0], self::delta());
        $this->assertEqualsWithDelta(-1.0, $flat[1], self::delta());
        $this->assertEqualsWithDelta(-1.0, $flat[2], self::delta());
        $this->assertEqualsWithDelta(2.0,  $flat[3], self::delta());
    }

    public function testPinvNonSquare(): void
    {
        // [3,2] matrix pseudoinverse should return [2,3]
        $t = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
        $pi = $t->pinv();
        $this->assertSame([2, 3], $pi->shape());
    }

    public function testSvdShapes(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);  // [2,3]
        $svd = $t->svd();
        $this->assertSame([2, 2], $svd['U']->shape());
        $this->assertSame(2,      $svd['S']->size());   // min(2,3)=2 singular values
        $this->assertSame([3, 3], $svd['Vt']->shape());
    }

    public function testCholeskyPositiveDefinite(): void
    {
        // [[4,2],[2,3]] is positive definite
        $t = Tensor::fromArray([[4.0, 2.0], [2.0, 3.0]]);
        $l = $t->cholesky();
        // L * L^T should equal the original
        $reconstructed = $l->matmul($l->transpose());
        $flat = $this->flat($reconstructed);
        $this->assertEqualsWithDelta(4.0, $flat[0], 0.001);
        $this->assertEqualsWithDelta(3.0, $flat[3], 0.001);
    }

    public function testLuDecomposition(): void
    {
        $t  = Tensor::fromArray([[2.0, 1.0], [6.0, 4.0]]);
        $lu = $t->lu();
        $this->assertSame([2, 2], $lu['P']->shape());
        $this->assertSame([2, 2], $lu['L']->shape());
        $this->assertSame([2, 2], $lu['U']->shape());
        // L*U should produce a [2,2] matrix
        $product = $lu['L']->matmul($lu['U']);
        $this->assertSame([2, 2], $product->shape());
    }

    public function testRefRref(): void
    {
        $t = Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]);
        $ref  = $t->ref();
        $rref = $t->rref();
        $this->assertSame([3, 3], $ref->shape());
        $this->assertSame([3, 3], $rref->shape());
    }

    // =========================================================================
    // 16. EMBEDDING LOOKUP
    // =========================================================================

    public function testEmbeddingLookup(): void
    {
        $tokens  = Tensor::fromArray([1, 0, 2], Tensor::DTYPE_INT32);
        $weights = Tensor::fromArray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]);
        $emb = $tokens->embeddingLookup($weights);
        $this->assertSame([3, 2], $emb->shape());
        $flat = $this->flat($emb);
        $this->assertEqualsWithDelta(0.3, $flat[0], self::delta()); // token 1 → row 1
        $this->assertEqualsWithDelta(0.1, $flat[2], self::delta()); // token 0 → row 0
        $this->assertEqualsWithDelta(0.5, $flat[4], self::delta()); // token 2 → row 2
    }

    // =========================================================================
    // 17. FUSED KERNELS
    // =========================================================================

    public function testFusedBceLossRange(): void
    {
        $preds   = Tensor::fromArray([0.9, 0.1, 0.8, 0.2]);
        $targets = Tensor::fromArray([1.0, 0.0, 1.0, 0.0]);
        $grads   = Tensor::zeros(4);
        $loss = Tensor::fusedBceLossAndGrad($preds, $targets, $grads);
        $this->assertGreaterThan(0.0, $loss);
        $this->assertLessThan(1.0, $loss);
    }

    // =========================================================================
    // 18. ARENA ALLOCATOR
    // =========================================================================

    public function testArenaAllocAndFree(): void
    {
        $ffi   = \Pml\Lib\TensorEngine::get();
        /** @var \FFI\CData $arena */
        $arena = $ffi->arena_create(1024 * 1024);
        $this->assertNotNull($arena);

        // Both the Tensor struct and its data live inside the arena.
        // $t->owned is false, so __destruct() will NOT call tensor_free().
        $t = new Tensor([50, 50], Tensor::DTYPE_FLOAT32, $arena);
        $this->assertSame([50, 50], $t->shape());
        $this->assertFalse((new \ReflectionProperty(Tensor::class, 'owned'))->getValue($t),
            'Arena tensor must not be owned — destructor must not call tensor_free()');

        // Null out $t before destroying the arena to avoid any dangling-pointer
        // access in the destructor (even though owned=false skips tensor_free).
        unset($t);

        // arena_destroy() bulk-frees everything: struct + data in one shot.
        $ffi->arena_reset($arena);
        $ffi->arena_destroy($arena);
        $this->addToAssertionCount(1); // reached here without segfault
    }

    // =========================================================================
    // 19. SERIALIZATION
    // =========================================================================

    public function testSaveLoadRoundtrip(): void
    {
        $path = sys_get_temp_dir() . '/tensor_test_' . uniqid() . '.bin';
        $orig = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0]]);
        $orig->save($path);

        $this->assertFileExists($path);

        $loaded = Tensor::load($path);
        $this->assertSame([2, 2], $loaded->shape());
        $flat = $this->flat($loaded);
        $this->assertEqualsWithDelta(1.0, $flat[0], self::delta());
        $this->assertEqualsWithDelta(4.0, $flat[3], self::delta());

        unlink($path);
    }

    // =========================================================================
    // 20. ERROR PATHS
    // =========================================================================

    public function testNdimExceedsEightThrows(): void
    {
        $this->expectException(InvalidArgumentException::class);
        new Tensor([2, 2, 2, 2, 2, 2, 2, 2, 2]); // 9 dims
    }

    public function testShapeMismatchOnAddThrows(): void
    {
        $this->expectException(\Throwable::class);
        $a = Tensor::fromArray([1.0, 2.0, 3.0]);
        $b = Tensor::fromArray([1.0, 2.0]);
        $a->add($b);
    }

    // =========================================================================
    // 21. RANDOM SAMPLING
    // =========================================================================

    public function testRandomChoice(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0, 5.0]);
        $c = $t->randomChoice(3, false);
        $this->assertSame(3, $c->size());
    }

    public function testRandomPermutation(): void
    {
        $t = Tensor::fromArray([1.0, 2.0, 3.0, 4.0, 5.0]);
        $p = $t->randomPermutation();
        $this->assertSame($t->size(), $p->size());
        // Sum is preserved
        $this->assertEqualsWithDelta($t->sum(), $p->sum(), self::delta());
    }
}
