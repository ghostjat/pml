<?php
require __DIR__.'/vendor/autoload.php';
use Pml\Blas;
use Pml\Matrix;
use Pml\Vector;
use Pml\ColumnVector;

function tensorTest(): void {
    $blas = Blas::getInstance();
    $ffiOk = $blas->available();
    $lapackOk = $blas->hasLapack();

    printf("═══════════════════════════════════════════════════\n");
    printf("  Tensor Library Self-Test\n");
    printf("═══════════════════════════════════════════════════\n");
    printf("  FFI/BLAS   : %s\n", $ffiOk ? '✓ available' : '✗ not found (pure PHP)');
    printf("  LAPACK     : %s\n", $lapackOk ? '✓ available' : '✗ not found (pure PHP fallbacks)');
    printf("───────────────────────────────────────────────────\n");

    $pass = 0;
    $fail = 0;
    $check = function (string $name, bool $ok, string $detail = '') use (&$pass, &$fail) {
        if ($ok) {
            printf("  ✓ %-40s\n", $name);
            ++$pass;
        } else {
            printf("  ✗ %-40s  %s\n", $name, $detail);
            ++$fail;
        }
    };

    // ── Matrix factories ──────────────────────────────────────────────

    $A = Matrix::build([[1, 2], [3, 4]]);
    $check('Matrix::build + shape', $A->shape() === [2, 2]);
    $check('Matrix::zeros', Matrix::zeros(3, 2)->sum() === 0.0);
    $check('Matrix::ones', Matrix::ones(2, 3)->sum() === 6.0);
    $check('Matrix::fill', Matrix::fill(5.0, 2, 2)->sum() === 20.0);
    $I = Matrix::eye(3);
    $check('Matrix::eye', $I->trace() === 3.0 && $I->det() === 1.0);
    $D = Matrix::diagonal([1, 2, 3]);
    $check('Matrix::diagonal', $D->trace() === 6.0);

    // ── Element ops ───────────────────────────────────────────────────

    $B = Matrix::build([[1, 4], [9, 16]]);
    $check('Matrix::sqrt', $B->sqrt()->asArray() === [[1.0, 2.0], [3.0, 4.0]]);
    $check('Matrix::square', Matrix::build([[1, 2], [3, 4]])->square()->asArray() === [[1.0, 4.0], [9.0, 16.0]]);
    $check('Matrix::negate', Matrix::build([[1, -2]])->negate()->asArray() === [[-1.0, 2.0]]);
    $check('Matrix::abs', Matrix::build([[-1, 2], [-3, 4]])->abs()->sum() === 10.0);
    $check('Matrix::clip', Matrix::build([[0, 5], [10, 15]])->clip(2, 8)->asArray() === [[2.0, 5.0], [8.0, 8.0]]);

    // ── Arithmetic ────────────────────────────────────────────────────

    $A = Matrix::build([[1, 2], [3, 4]]);
    $check('Matrix add scalar', $A->add(1)->sum() === 14.0);
    $check('Matrix sub scalar', $A->subtract(1)->sum() === 6.0);
    $check('Matrix mul scalar', $A->multiply(2)->sum() === 20.0);
    $check('Matrix div scalar', $A->divide(2)->asArray() === [[0.5, 1.0], [1.5, 2.0]]);
    $A2 = Matrix::build([[1, 0], [0, 1]]);
    $check('Matrix add matrix', $A->add($A2)->asArray() === [[2.0, 2.0], [3.0, 5.0]]);

    // ── Reductions ────────────────────────────────────────────────────

    $A = Matrix::build([[1, 2, 3], [4, 5, 6]]);
    $check('rowSums', $A->rowSums()->asArray() === [6.0, 15.0]);
    $check('colSums', $A->columnSums()->asArray() === [5.0, 7.0, 9.0]);
    $check('rowMeans', $A->rowMeans()->asArray() === [2.0, 5.0]);
    $check('colMeans', $A->columnMeans()->asArray() === [2.5, 3.5, 4.5]);
    $check('rowMaxima', $A->rowMaxima()->asArray() === [3.0, 6.0]);
    $check('rowMinima', $A->rowMinima()->asArray() === [1.0, 4.0]);

    // ── Matmul ────────────────────────────────────────────────────────

    $A = Matrix::build([[1, 2], [3, 4]]);
    $B = Matrix::build([[5, 6], [7, 8]]);
    $C = $A->matmul($B);
    $expected = [[19.0, 22.0], [43.0, 50.0]];
    $ok = true;
    foreach ($C->asArray() as $i => $row)
        foreach ($row as $j => $v)
            if (abs($v - $expected[$i][$j]) > 1e-9)
                $ok = false;
    $check('Matrix::matmul 2×2', $ok, sprintf("got %s", $C));

    // ── Matmul large (benchmark) ──────────────────────────────────────

    $n = 512;
    $M1 = Matrix::rand($n, $n);
    $M2 = Matrix::rand($n, $n);
    $t0 = microtime(true);
    $M3 = $M1->matmul($M2);
    $tMM = microtime(true) - $t0;
    $check("matmul {$n}×{$n} (using " . ($ffiOk ? 'BLAS' : 'PHP') . ')', $M3->m() === $n && $M3->n() === $n);
    printf("    time: %.3f sec\n", $tMM);

    // ── Transpose ────────────────────────────────────────────────────

    $T = Matrix::build([[1, 2, 3], [4, 5, 6]])->transpose();
    $check('Matrix::transpose', $T->shape() === [3, 2] && $T->asArray()[0] === [1.0, 4.0]);

    // ── Dot (matrix × vector) ─────────────────────────────────────────

    $A = Matrix::build([[1, 2], [3, 4]]);
    $x = Vector::build([1, 1]);
    $y = $A->dot($x);
    $check('Matrix::dot(Vector)', $y instanceof ColumnVector && $y->asArray() === [3.0, 7.0]);

    // ── LU ───────────────────────────────────────────────────────────

    $A = Matrix::build([[2, 1, 1], [4, 3, 3], [8, 7, 9]]);
    ['L' => $L, 'U' => $U, 'P' => $P] = $A->lu();
    $PA = $P->matmul($A);
    $LU = $L->matmul($U);
    $ok = true;
    foreach ($PA->asArray() as $i => $row)
        foreach ($row as $j => $v)
            if (abs($v - $LU->asArray()[$i][$j]) > 1e-8)
                $ok = false;
    $check('Matrix::lu  P·A = L·U', $ok);

    // ── Inverse ──────────────────────────────────────────────────────

    $A = Matrix::build([[2, 1], [5, 3]]);
    $inv = $A->inverse();
    $I2 = $A->matmul($inv);
    $ok = abs($I2->asArray()[0][0] - 1.0) < 1e-9 && abs($I2->asArray()[0][1]) < 1e-9;
    $check('Matrix::inverse  A·A⁻¹ = I', $ok);

    // ── Det ──────────────────────────────────────────────────────────

    $check('Matrix::det 2×2', abs(Matrix::build([[3, 8], [4, 6]])->det() - (-14.0)) < 1e-9);

    // ── Solve ─────────────────────────────────────────────────────────

    $A = Matrix::build([[2, 1], [-1, 3]]);
    $b = Matrix::build([[5], [0]]);
    $x = $A->solve($b);
    $ok = abs($x->asArray()[0][0] - 3.0) < 1e-8 && abs($x->asArray()[1][0] - (-1.0)) < 1e-8;
    $check('Matrix::solve  Ax=b', $ok, sprintf("x=%s", $x));

    // ── Cholesky ──────────────────────────────────────────────────────

    $A = Matrix::build([[4, 2], [2, 3]]);
    $L = $A->cholesky();
    $LLt = $L->matmul($L->transpose());
    $ok = abs($LLt->asArray()[0][0] - 4.0) < 1e-9 && abs($LLt->asArray()[1][0] - 2.0) < 1e-9;
    $check('Matrix::cholesky  L·Lᵀ = A', $ok);

    // ── SVD ───────────────────────────────────────────────────────────

    $A = Matrix::build([[1, 0, 0], [0, 2, 0], [0, 0, 3]]);
    ['U' => $U, 's' => $s, 'VT' => $VT] = $A->svd();
    $sv = $s->asArray();
    rsort($sv);
    $check('Matrix::svd singular values', abs($sv[0] - 3.0) < 1e-6 && abs($sv[1] - 2.0) < 1e-6 && abs($sv[2] - 1.0) < 1e-6);

    // ── Eig (symmetric) ───────────────────────────────────────────────

    $A = Matrix::build([[2, -1], [-1, 2]]);
    ['values' => $ev] = $A->eig(true);
    $vals = $ev->asArray();
    sort($vals);
    $check('Matrix::eig symmetric eigenvalues', abs($vals[0] - 1.0) < 1e-8 && abs($vals[1] - 3.0) < 1e-8);

    // ── REF / RREF ────────────────────────────────────────────────────

    $A = Matrix::build([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    $check('Matrix::rank (singular)', $A->rank() === 2);
    $R = $A->rref();
    $check('Matrix::rref top-left = 1', abs($R->asArray()[0][0] - 1.0) < 1e-9);

    // ── Vector ops ────────────────────────────────────────────────────

    $v1 = Vector::build([3, 4]);
    $check('Vector::norm L2', abs($v1->norm() - 5.0) < 1e-10);
    $check('Vector::norm L1', abs($v1->norm(1.0) - 7.0) < 1e-10);
    $v2 = Vector::build([1, 0]);
    $check('Vector::dot', abs($v1->dot($v2) - 3.0) < 1e-10);
    $check('Vector::normalize', abs($v1->normalize()->norm() - 1.0) < 1e-10);

    $v3 = Vector::build([1, 0, 0]);
    $v4 = Vector::build([0, 1, 0]);
    $cross = $v3->cross($v4);
    $check('Vector::cross', $cross->asArray() === [0.0, 0.0, 1.0]);

    $outer = $v3->outer($v4);
    $check('Vector::outer shape', $outer->shape() === [3, 3]);

    $cv = ColumnVector::build([1, 2, 3]);
    $check('ColumnVector instance', $cv instanceof ColumnVector && $cv instanceof Vector);
    $check('ColumnVector::asMatrix', $cv->asMatrix()->shape() === [3, 1]);

    // ── Softmax ───────────────────────────────────────────────────────

    $A = Matrix::build([[1.0, 2.0, 3.0]]);
    $sm = $A->softmax()->asArray()[0];
    $check('Matrix::softmax sums to 1', abs(array_sum($sm) - 1.0) < 1e-12);

    // ── Comparisons ───────────────────────────────────────────────────

    $v = Vector::build([1, 2, 3, 4, 5]);
    $check('Vector::greater scalar', $v->greater(3)->asArray() === [0.0, 0.0, 0.0, 1.0, 1.0]);
    $check('Vector::lessEqual scalar', $v->lessEqual(3)->asArray() === [1.0, 1.0, 1.0, 0.0, 0.0]);

    // ── Summary ───────────────────────────────────────────────────────

    printf("───────────────────────────────────────────────────\n");
    printf("  Passed: %d / %d\n", $pass, $pass + $fail);
    if ($fail > 0)
        printf("  FAILED: %d\n", $fail);
    printf("═══════════════════════════════════════════════════\n");
}

tensorTest();
