<?php
declare(strict_types=1);

/**
 * Tensor — FFI-backed linear algebra for PHP
 * ============================================
 * Drop-in compatible with RubixML/Tensor (same namespace, same signatures).
 *
 * ── INTEGRATION ───────────────────────────────────────────────────────────
 *
 * Option A — standalone (no Composer):
 *   require_once __DIR__ . '/Tensor.php';
 *   use Tensor\Matrix;
 *   $C = Matrix::rand(128, 128)->matmul(Matrix::rand(128, 128));
 *
 * Option B — alongside RubixML in composer.json:
 *   If rubix/tensor is already required, do NOT require this file too.
 *   This file IS the rubix/tensor replacement; use one or the other.
 *   To replace: remove "rubix/tensor" from composer.json and add
 *     "files": ["path/to/Tensor.php"]
 *   in the "autoload" section.
 *
 * Option C — alongside your own Pml\Matrix etc:
 *   This file uses namespace Tensor\, not Pml\. There is NO naming conflict
 *   as long as this file is only included ONCE. The include guard
 *   (_RUBIX_TENSOR_FFI_LOADED_) prevents double-loading.
 *   If you see "Cannot redeclare", something is including this file twice
 *   or an autoloader is interfering. Fix: use require_once, not require.
 *
 * ── ENABLING FFI/BLAS ─────────────────────────────────────────────────────
 *
 *   1. Install OpenBLAS:   sudo apt install libopenblas-dev
 *   2. Enable FFI in php.ini:
 *        ffi.enable = true          ; development
 *        ffi.enable = preload       ; production (add preload_user too)
 *   3. Verify: php -r "echo extension_loaded('ffi') ? 'ok' : 'no';"
 *
 * ── BACKEND SELECTION (automatic, per-operation) ──────────────────────────
 *
 *   BLAS (cblas_dgemm, cblas_dgemv, cblas_ddot, cblas_dnrm2):
 *     matmul, Matrix::dot, Vector::dot, Vector::norm(2)
 *   LAPACK (LAPACKE_* C interface):
 *     inverse, solve, svd, eig, cholesky
 *   Pure PHP fallback for everything when FFI unavailable:
 *     Tiled blocked multiply (BLOCK=32, 4× unrolled), LU, Cholesky,
 *     Jacobi eigendecomposition, Golub-Kahan SVD, QR, REF/RREF
 *
 * ── STORAGE ───────────────────────────────────────────────────────────────
 *
 *   PHP arrays as canonical storage (serialisable, parallel\Future-safe).
 *   FFI double[] buffers are allocated transiently per BLAS call, freed
 *   immediately after the result is copied back. No persistent FFI state.
 *
 * ── SELF-TEST ─────────────────────────────────────────────────────────────
 *
 *   php Tensor.php          — runs 35 assertions, shows BLAS/LAPACK status
 */

namespace Tensor;

// ── Include guard — safe to require_once or include multiple times ─────────
// Uses a global constant with a collision-resistant name.
if (defined('_RUBIX_TENSOR_FFI_LOADED_')) return;
define('_RUBIX_TENSOR_FFI_LOADED_', true);


// ═══════════════════════════════════════════════════════════════════════════
//  BLAS / LAPACK FFI BACKEND
// ═══════════════════════════════════════════════════════════════════════════

final class Blas
{
    private static ?self $instance = null;
    private ?\FFI $ffi = null;
    private bool  $hasLapack = false;

    /** Runtime-discovered library paths, broadest possible search */
    private static function candidates(): array
    {
        $fixed = [
            // OpenBLAS pthread variant (most common on Ubuntu/Debian)
            '/usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblas.so.0',
            '/usr/lib/x86_64-linux-gnu/openblas-serial/libopenblas.so.0',
            '/usr/lib/x86_64-linux-gnu/libopenblas.so.0',
            '/usr/lib/aarch64-linux-gnu/openblas-pthread/libopenblas.so.0',
            '/usr/lib/aarch64-linux-gnu/libopenblas.so.0',
            // cblas shim
            '/usr/lib/x86_64-linux-gnu/openblas-pthread/libcblas.so.3',
            '/usr/lib/x86_64-linux-gnu/libcblas.so.3',
            // plain BLAS (reference or ATLAS)
            '/usr/lib/x86_64-linux-gnu/libblas.so.3',
            '/usr/lib/libblas.so.3',
            // macOS Accelerate
            '/System/Library/Frameworks/Accelerate.framework/Accelerate',
            // short names (ldconfig path)
            'libopenblas.so.0', 'libopenblas.so',
            'libcblas.so.3',    'libcblas.so',
            'libblas.so.3',     'libblas.so',
        ];
        // glob for any version number variant not listed above
        $globs = [
            '/usr/lib/x86_64-linux-gnu/openblas*/libopenblas*.so*',
            '/usr/lib/x86_64-linux-gnu/libopenblas*.so*',
            '/usr/lib/x86_64-linux-gnu/libcblas*.so*',
            '/usr/lib/*/libopenblas*.so*',
        ];
        $found = [];
        foreach ($globs as $g) {
            $matches = glob($g) ?: [];
            foreach ($matches as $path) {
                if (!in_array($path, $fixed, true)) $found[] = $path;
            }
        }
        return array_merge($fixed, $found);
    }

    // BLAS+LAPACK combined header (LAPACKE C interface)
    private const HEADER = '
        /* BLAS enums */
        typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_LAYOUT;
        typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;
        typedef enum { CblasUpper=121, CblasLower=122 } CBLAS_UPLO;
        typedef enum { CblasNonUnit=131, CblasUnit=132 } CBLAS_DIAG;
        typedef enum { CblasLeft=141, CblasRight=142 } CBLAS_SIDE;

        /* Level 1 BLAS */
        double cblas_ddot (int n, const double *x, int incx, const double *y, int incy);
        double cblas_dnrm2(int n, const double *x, int incx);
        double cblas_dasum(int n, const double *x, int incx);
        int    cblas_idamax(int n, const double *x, int incx);
        void   cblas_dscal(int n, double alpha, double *x, int incx);
        void   cblas_dcopy(int n, const double *x, int incx, double *y, int incy);
        void   cblas_daxpy(int n, double alpha, const double *x, int incx, double *y, int incy);
        void   cblas_dswap(int n, double *x, int incx, double *y, int incy);

        /* Level 2 BLAS */
        void cblas_dgemv(int Order, int TransA, int M, int N,
                         double alpha, const double *A, int lda,
                         const double *x, int incx,
                         double beta,  double *y, int incy);

        void cblas_dsymv(int Order, int Uplo, int N,
                         double alpha, const double *A, int lda,
                         const double *x, int incx,
                         double beta,  double *y, int incy);

        /* Level 3 BLAS */
        void cblas_dgemm(int Order, int TransA, int TransB,
                         int M, int N, int K,
                         double alpha, const double *A, int lda,
                         const double *B, int ldb,
                         double beta,  double *C, int ldc);

        /* LAPACKE (C interface to LAPACK) */
        int LAPACKE_dgetrf(int matrix_layout, int m, int n,
                           double *a, int lda, int *ipiv);
        int LAPACKE_dgetri(int matrix_layout, int n,
                           double *a, int lda, const int *ipiv);
        int LAPACKE_dgetrs(int matrix_layout, char trans, int n, int nrhs,
                           const double *a, int lda, const int *ipiv,
                           double *b, int ldb);
        int LAPACKE_dgesv (int matrix_layout, int n, int nrhs,
                           double *a, int lda, int *ipiv,
                           double *b, int ldb);
        int LAPACKE_dgesvd(int matrix_layout, char jobu, char jobvt,
                           int m, int n, double *a, int lda,
                           double *s, double *u, int ldu,
                           double *vt, int ldvt, double *superb);
        int LAPACKE_dsyev (int matrix_layout, char jobz, char uplo,
                           int n, double *a, int lda, double *w);
        int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr,
                           int n, double *a, int lda,
                           double *wr, double *wi,
                           double *vl, int ldvl,
                           double *vr, int ldvr);
        int LAPACKE_dpotrf(int matrix_layout, char uplo, int n,
                           double *a, int lda);
        int LAPACKE_dgels (int matrix_layout, char trans, int m, int n,
                           int nrhs, double *a, int lda, double *b, int ldb);
        int LAPACKE_dgeqrf(int matrix_layout, int m, int n,
                           double *a, int lda, double *tau);
        int LAPACKE_dorgqr(int matrix_layout, int m, int n, int k,
                           double *a, int lda, const double *tau);
        int LAPACKE_dtrtrs(int matrix_layout, char uplo, char trans, char diag,
                           int n, int nrhs, const double *a, int lda,
                           double *b, int ldb);
    ';

    private function __construct()
    {
        if (!extension_loaded('ffi')) return;
        foreach (self::candidates() as $lib) {
            try {
                $this->ffi = \FFI::cdef(self::HEADER, $lib);
                // probe for LAPACK by checking symbol existence via calling a trivial op
                try {
                    // LAPACKE_dpotrf on 1×1 identity — safe probe
                    $a = $this->ffi->new('double[1]');
                    $a[0] = 1.0;
                    $rc = $this->ffi->LAPACKE_dpotrf(101, 'U', 1, $a, 1);
                    $this->hasLapack = ($rc === 0);
                } catch (\Throwable) {
                    $this->hasLapack = false;
                }
                return;
            } catch (\Throwable) {}
        }
    }

    public static function getInstance(): self
    {
        return self::$instance ??= new self();
    }

    public function available(): bool   { return $this->ffi !== null; }
    public function hasLapack(): bool   { return $this->hasLapack; }
    public function ffi(): ?\FFI        { return $this->ffi; }

    // ── buffer helpers ────────────────────────────────────────────────────

    /** Flat PHP array → native double[] */
    public function toBuffer(array $data): \FFI\CData
    {
        $n   = count($data);
        $buf = $this->ffi->new("double[$n]");
        for ($i = 0; $i < $n; ++$i) $buf[$i] = $data[$i];
        return $buf;
    }

    /** 2-D PHP array-of-arrays → row-major native double[] */
    public function matToBuffer(array $rows, int $m, int $n): \FFI\CData
    {
        $buf = $this->ffi->new("double[".($m*$n)."]");
        $k   = 0;
        foreach ($rows as $row) {
            foreach ($row as $v) $buf[$k++] = $v;
        }
        return $buf;
    }

    /** native double[n] → flat PHP array */
    public function fromBuffer(\FFI\CData $buf, int $n): array
    {
        $a = [];
        for ($i = 0; $i < $n; ++$i) $a[$i] = $buf[$i];
        return $a;
    }

    /** native double[m*n] → array-of-arrays [m][n] */
    public function bufToMatrix(\FFI\CData $buf, int $m, int $n): array
    {
        $rows = [];
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            $off = $i * $n;
            for ($j = 0; $j < $n; ++$j) $row[$j] = $buf[$off + $j];
            $rows[$i] = $row;
        }
        return $rows;
    }

    // ── BLAS wrappers ─────────────────────────────────────────────────────

    /** C = alpha*A*B + beta*C  (row-major, A:m×k, B:k×n, C:m×n) */
    public function dgemm(
        array $A, int $m, int $k,
        array $B, int $n,
        float $alpha = 1.0, float $beta = 0.0
    ): array {
        $bA = $this->matToBuffer($A, $m, $k);
        $bB = $this->matToBuffer($B, $k, $n);
        $bC = $this->ffi->new("double[".($m*$n)."]");
        // CblasRowMajor=101, CblasNoTrans=111
        $this->ffi->cblas_dgemm(101, 111, 111, $m, $n, $k,
            $alpha, $bA, $k, $bB, $n, $beta, $bC, $n);
        return $this->bufToMatrix($bC, $m, $n);
    }

    /** y = alpha*A*x + beta*y  (A:m×n, x:n, y:m) */
    public function dgemv(array $A, int $m, int $n, array $x,
                          float $alpha=1.0, float $beta=0.0): array
    {
        $bA = $this->matToBuffer($A, $m, $n);
        $bx = $this->toBuffer($x);
        $by = $this->ffi->new("double[$m]");
        $this->ffi->cblas_dgemv(101, 111, $m, $n, $alpha, $bA, $n,
                                 $bx, 1, $beta, $by, 1);
        return $this->fromBuffer($by, $m);
    }

    /** dot product of two flat arrays */
    public function ddot(array $x, array $y, int $n): float
    {
        $bx = $this->toBuffer($x);
        $by = $this->toBuffer($y);
        return $this->ffi->cblas_ddot($n, $bx, 1, $by, 1);
    }

    /** L2 norm */
    public function dnrm2(array $x, int $n): float
    {
        $bx = $this->toBuffer($x);
        return $this->ffi->cblas_dnrm2($n, $bx, 1);
    }

    /** sum of absolute values */
    public function dasum(array $x, int $n): float
    {
        $bx = $this->toBuffer($x);
        return $this->ffi->cblas_dasum($n, $bx, 1);
    }

    // ── LAPACK wrappers ───────────────────────────────────────────────────

    /**
     * LU factorisation: returns [L, U, P, ipiv_array]
     * A is row-major m×n array-of-arrays
     */
    public function dgetrf(array $A, int $m, int $n): array
    {
        $buf  = $this->matToBuffer($A, $m, $n);
        $ipiv = $this->ffi->new("int[".min($m,$n)."]");
        $rc   = $this->ffi->LAPACKE_dgetrf(101, $m, $n, $buf, $n, $ipiv);
        if ($rc !== 0) throw new \RuntimeException("dgetrf failed: info=$rc");
        return [$buf, $ipiv];
    }

    /**
     * Matrix inverse via LU: returns array-of-arrays
     */
    public function inverse(array $A, int $n): array
    {
        [$buf, $ipiv] = $this->dgetrf($A, $n, $n);
        $rc = $this->ffi->LAPACKE_dgetri(101, $n, $buf, $n, $ipiv);
        if ($rc !== 0) throw new \RuntimeException("dgetri failed: info=$rc");
        return $this->bufToMatrix($buf, $n, $n);
    }

    /**
     * Solve A*X = B; A is n×n, B is n×nrhs
     * Returns X as array-of-arrays (n×nrhs)
     */
    public function solve(array $A, array $B, int $n, int $nrhs): array
    {
        $bA   = $this->matToBuffer($A, $n, $n);
        $bB   = $this->matToBuffer($B, $n, $nrhs);
        $ipiv = $this->ffi->new("int[$n]");
        $rc   = $this->ffi->LAPACKE_dgesv(101, $n, $nrhs, $bA, $n, $ipiv, $bB, $nrhs);
        if ($rc !== 0) throw new \RuntimeException("dgesv failed: info=$rc");
        return $this->bufToMatrix($bB, $n, $nrhs);
    }

    /**
     * SVD: A (m×n) → [U(m×m), s(min(m,n)), VT(n×n)]
     */
    public function dgesvd(array $A, int $m, int $n): array
    {
        $k    = min($m, $n);
        $bA   = $this->matToBuffer($A, $m, $n);
        $bS   = $this->ffi->new("double[$k]");
        $bU   = $this->ffi->new("double[".($m*$m)."]");
        $bVT  = $this->ffi->new("double[".($n*$n)."]");
        $sup  = $this->ffi->new("double[$k]");
        $rc   = $this->ffi->LAPACKE_dgesvd(101, 'A', 'A', $m, $n,
                    $bA, $n, $bS, $bU, $m, $bVT, $n, $sup);
        if ($rc !== 0) throw new \RuntimeException("dgesvd failed: info=$rc");
        return [
            $this->bufToMatrix($bU, $m, $m),
            $this->fromBuffer($bS, $k),
            $this->bufToMatrix($bVT, $n, $n),
        ];
    }

    /**
     * Eigendecomposition (symmetric): A (n×n) → [eigenvalues, eigenvectors]
     */
    public function dsyev(array $A, int $n): array
    {
        $buf = $this->matToBuffer($A, $n, $n);
        $w   = $this->ffi->new("double[$n]");
        $rc  = $this->ffi->LAPACKE_dsyev(101, 'V', 'U', $n, $buf, $n, $w);
        if ($rc !== 0) throw new \RuntimeException("dsyev failed: info=$rc");
        return [$this->fromBuffer($w, $n), $this->bufToMatrix($buf, $n, $n)];
    }

    /**
     * Eigendecomposition (general): A (n×n) → [real_parts, imag_parts, right_vecs]
     */
    public function dgeev(array $A, int $n): array
    {
        $buf = $this->matToBuffer($A, $n, $n);
        $wr  = $this->ffi->new("double[$n]");
        $wi  = $this->ffi->new("double[$n]");
        $vl  = $this->ffi->new("double[1]");   // not needed
        $vr  = $this->ffi->new("double[".($n*$n)."]");
        $rc  = $this->ffi->LAPACKE_dgeev(101, 'N', 'V', $n, $buf, $n,
                   $wr, $wi, $vl, 1, $vr, $n);
        if ($rc !== 0) throw new \RuntimeException("dgeev failed: info=$rc");
        return [
            $this->fromBuffer($wr, $n),
            $this->fromBuffer($wi, $n),
            $this->bufToMatrix($vr, $n, $n),
        ];
    }

    /**
     * Cholesky: A (n×n symmetric positive definite) → lower-triangular L
     */
    public function dpotrf(array $A, int $n): array
    {
        $buf = $this->matToBuffer($A, $n, $n);
        $rc  = $this->ffi->LAPACKE_dpotrf(101, 'L', $n, $buf, $n);
        if ($rc !== 0) throw new \RuntimeException("dpotrf failed: info=$rc (matrix not positive definite)");
        // Zero out upper triangle (LAPACK leaves it untouched)
        $rows = $this->bufToMatrix($buf, $n, $n);
        for ($i = 0; $i < $n; ++$i)
            for ($j = $i+1; $j < $n; ++$j)
                $rows[$i][$j] = 0.0;
        return $rows;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  TENSOR INTERFACE
// ═══════════════════════════════════════════════════════════════════════════

interface Tensor extends \Countable, \IteratorAggregate, \JsonSerializable
{
    // dtype constants (match RubixML)
    public const FLOAT = 1;
    public const INT   = 2;

    public function shape(): array;
    public function shapeString(): string;
    public function isSquare(): bool;
    public function size(): int;
    public function ndim(): int;
    public function dtype(): int;
    public function asArray(): array;
    public function flatten(): Vector;

    // reductions
    public function sum(): int|float;
    public function product(): int|float;
    public function min(): int|float;
    public function max(): int|float;
    public function mean(): float;
    public function median(): float;
    public function variance(?Tensor $mean = null): float;

    // element-wise ops (return same type)
    public function round(int $precision = 0): static;
    public function floor(): static;
    public function ceil(): static;
    public function clip(float $min, float $max): static;
    public function clipLower(float $min): static;
    public function clipUpper(float $max): static;
    public function abs(): static;
    public function square(): static;
    public function sqrt(): static;
    public function exp(): static;
    public function expm1(): static;
    public function log(float $base = M_E): static;
    public function log1p(): static;
    public function sin(): static;
    public function cos(): static;
    public function tan(): static;
    public function asin(): static;
    public function acos(): static;
    public function atan(): static;
    public function sinh(): static;
    public function cosh(): static;
    public function tanh(): static;
    public function negate(): static;
    public function reciprocal(): static;

    // arithmetic (scalar or Tensor argument)
    public function add(int|float|Tensor $b): static;
    public function subtract(int|float|Tensor $b): static;
    public function multiply(int|float|Tensor $b): static;
    public function divide(int|float|Tensor $b): static;
    public function pow(int|float|Tensor $b): static;
    public function mod(int|float|Tensor $b): static;

    // comparisons (return same shape, 1.0 / 0.0)
    public function equal(int|float|Tensor $b): static;
    public function notEqual(int|float|Tensor $b): static;
    public function greater(int|float|Tensor $b): static;
    public function greaterEqual(int|float|Tensor $b): static;
    public function less(int|float|Tensor $b): static;
    public function lessEqual(int|float|Tensor $b): static;

    // functional
    public function map(callable $fn): static;
    public function reduce(callable $fn, int|float $initial = 0): int|float;
}

// ═══════════════════════════════════════════════════════════════════════════
//  PURE-PHP LINEAR ALGEBRA UTILITIES
//  (Used as fallback when FFI/LAPACK unavailable)
// ═══════════════════════════════════════════════════════════════════════════

final class LinAlg
{
    private const BLOCK = 32;

    // ── matrix multiply (tiled, 4× unrolled) ─────────────────────────────

    /** Multiply A(m×k) · B(k×n), both as array-of-arrays. Returns array-of-arrays. */
    public static function matmul(array $A, array $B, int $m, int $k, int $n): array
    {
        // Transpose B for cache-friendly access
        $BT = self::transposeRaw($B, $k, $n);
        $C  = array_fill(0, $m, null);
        for ($i = 0; $i < $m; ++$i) $C[$i] = array_fill(0, $n, 0.0);

        $bl = self::BLOCK;
        for ($ii = 0; $ii < $m; $ii += $bl) {
            $iMx = min($ii + $bl, $m);
            for ($jj = 0; $jj < $n; $jj += $bl) {
                $jMx = min($jj + $bl, $n);
                for ($kk = 0; $kk < $k; $kk += $bl) {
                    $kMx  = min($kk + $bl, $k);
                    $kMx4 = $kk + (int)(($kMx - $kk) / 4) * 4;
                    for ($i = $ii; $i < $iMx; ++$i) {
                        $Ar = $A[$i];
                        $Cr = &$C[$i];
                        for ($j = $jj; $j < $jMx; ++$j) {
                            $BTr = $BT[$j];
                            $s0 = $s1 = $s2 = $s3 = 0.0;
                            for ($kp = $kk; $kp < $kMx4; $kp += 4) {
                                $s0 += $Ar[$kp]   * $BTr[$kp];
                                $s1 += $Ar[$kp+1] * $BTr[$kp+1];
                                $s2 += $Ar[$kp+2] * $BTr[$kp+2];
                                $s3 += $Ar[$kp+3] * $BTr[$kp+3];
                            }
                            $Cr[$j] += $s0 + $s1 + $s2 + $s3;
                            for ($kp = $kMx4; $kp < $kMx; ++$kp)
                                $Cr[$j] += $Ar[$kp] * $BTr[$kp];
                        }
                    }
                }
            }
        }
        return $C;
    }

    /** Matrix × vector: A(m×n) · x(n) → y(m) flat array */
    public static function matvec(array $A, array $x, int $m, int $n): array
    {
        $y = array_fill(0, $m, 0.0);
        for ($i = 0; $i < $m; ++$i) {
            $s = 0.0;
            $r = $A[$i];
            for ($j = 0; $j < $n; ++$j) $s += $r[$j] * $x[$j];
            $y[$i] = $s;
        }
        return $y;
    }

    // ── transpose ─────────────────────────────────────────────────────────

    /** Transpose m×n array-of-arrays → n×m */
    public static function transposeRaw(array $M, int $m, int $n): array
    {
        $T = [];
        for ($j = 0; $j < $n; ++$j) $T[$j] = array_fill(0, $m, 0.0);
        for ($i = 0; $i < $m; ++$i)
            for ($j = 0; $j < $n; ++$j)
                $T[$j][$i] = $M[$i][$j];
        return $T;
    }

    // ── LU decomposition (in-place, partial pivoting) ─────────────────────

    /**
     * Returns [L_and_U_combined, pivot_indices]
     * L is unit lower-triangular (diagonal 1s stored implicitly),
     * U is upper-triangular.
     * Both stored in the same n×n array (standard LAPACK layout).
     */
    public static function lu(array $A, int $n): array
    {
        $LU   = $A;  // copy
        $ipiv = array_fill(0, $n, 0);

        for ($k = 0; $k < $n; ++$k) {
            // find pivot
            $maxVal = abs($LU[$k][$k]);
            $maxRow = $k;
            for ($i = $k+1; $i < $n; ++$i) {
                if (($v = abs($LU[$i][$k])) > $maxVal) { $maxVal = $v; $maxRow = $i; }
            }
            $ipiv[$k] = $maxRow;
            if ($maxRow !== $k) {  // swap rows
                [$LU[$k], $LU[$maxRow]] = [$LU[$maxRow], $LU[$k]];
            }
            if (abs($LU[$k][$k]) < 1e-15) continue;  // singular
            $invPivot = 1.0 / $LU[$k][$k];
            for ($i = $k+1; $i < $n; ++$i) {
                $LU[$i][$k] *= $invPivot;
                $lik = $LU[$i][$k];
                for ($j = $k+1; $j < $n; ++$j)
                    $LU[$i][$j] -= $lik * $LU[$k][$j];
            }
        }
        return [$LU, $ipiv];
    }

    /** Determinant from LU factorization */
    public static function det(array $A, int $n): float
    {
        [$LU, $ipiv] = self::lu($A, $n);
        $det  = 1.0;
        $swaps = 0;
        for ($i = 0; $i < $n; ++$i) {
            $det *= $LU[$i][$i];
            if ($ipiv[$i] !== $i) ++$swaps;
        }
        return ($swaps % 2 === 0) ? $det : -$det;
    }

    /** Solve A*x = b using pre-computed LU. b is a flat array. */
    public static function luSolve(array $LU, array $ipiv, array $b, int $n): array
    {
        $x = $b;
        // apply pivots
        for ($i = 0; $i < $n; ++$i)
            if ($ipiv[$i] !== $i) [$x[$i], $x[$ipiv[$i]]] = [$x[$ipiv[$i]], $x[$i]];
        // forward substitution (L is unit lower-triangular)
        for ($i = 1; $i < $n; ++$i) {
            for ($j = 0; $j < $i; ++$j)
                $x[$i] -= $LU[$i][$j] * $x[$j];
        }
        // back substitution (U)
        for ($i = $n-1; $i >= 0; --$i) {
            for ($j = $i+1; $j < $n; ++$j)
                $x[$i] -= $LU[$i][$j] * $x[$j];
            $x[$i] /= $LU[$i][$i];
        }
        return $x;
    }

    /** Inverse via LU */
    public static function inverse(array $A, int $n): array
    {
        [$LU, $ipiv] = self::lu($A, $n);
        $inv = [];
        for ($j = 0; $j < $n; ++$j) {
            $e = array_fill(0, $n, 0.0);
            $e[$j] = 1.0;
            $col = self::luSolve($LU, $ipiv, $e, $n);
            for ($i = 0; $i < $n; ++$i) $inv[$i][$j] = $col[$i];
        }
        return $inv;
    }

    /** Solve A*X = B where B is array-of-arrays (n×nrhs) */
    public static function solve(array $A, array $B, int $n, int $nrhs): array
    {
        [$LU, $ipiv] = self::lu($A, $n);
        $X = [];
        for ($j = 0; $j < $nrhs; ++$j) {
            $b = array_column($B, $j);
            $x = self::luSolve($LU, $ipiv, $b, $n);
            for ($i = 0; $i < $n; ++$i) $X[$i][$j] = $x[$i];
        }
        return $X;
    }

    // ── Cholesky decomposition ─────────────────────────────────────────────

    /** Cholesky-Banachiewicz: A = L*L^T, returns L (lower triangular) */
    public static function cholesky(array $A, int $n): array
    {
        $L = array_fill(0, $n, null);
        for ($i = 0; $i < $n; ++$i) $L[$i] = array_fill(0, $n, 0.0);

        for ($i = 0; $i < $n; ++$i) {
            for ($j = 0; $j <= $i; ++$j) {
                $s = $A[$i][$j];
                for ($k = 0; $k < $j; ++$k) $s -= $L[$i][$k] * $L[$j][$k];
                if ($i === $j) {
                    if ($s < 0.0) throw new \RuntimeException("Matrix is not positive definite");
                    $L[$i][$j] = sqrt($s);
                } else {
                    $L[$i][$j] = $s / $L[$j][$j];
                }
            }
        }
        return $L;
    }

    // ── QR decomposition (Gram-Schmidt) ────────────────────────────────────

    /** Returns [Q(m×n), R(n×n)] via modified Gram-Schmidt */
    public static function qr(array $A, int $m, int $n): array
    {
        $Q = $A;
        $R = array_fill(0, $n, null);
        for ($i = 0; $i < $n; ++$i) $R[$i] = array_fill(0, $n, 0.0);

        for ($j = 0; $j < $n; ++$j) {
            // extract column j of Q
            $v = [];
            for ($i = 0; $i < $m; ++$i) $v[$i] = $Q[$i][$j];
            // orthogonalise against previous columns
            for ($k = 0; $k < $j; ++$k) {
                $q = [];
                for ($i = 0; $i < $m; ++$i) $q[$i] = $Q[$i][$k];
                $dot = 0.0;
                for ($i = 0; $i < $m; ++$i) $dot += $q[$i] * $v[$i];
                $R[$k][$j] = $dot;
                for ($i = 0; $i < $m; ++$i) $v[$i] -= $dot * $q[$i];
            }
            $norm = 0.0;
            for ($i = 0; $i < $m; ++$i) $norm += $v[$i] * $v[$i];
            $norm = sqrt($norm);
            $R[$j][$j] = $norm;
            if ($norm > 1e-14)
                for ($i = 0; $i < $m; ++$i) $Q[$i][$j] = $v[$i] / $norm;
            else
                for ($i = 0; $i < $m; ++$i) $Q[$i][$j] = 0.0;
        }
        return [$Q, $R];
    }

    // ── SVD (Golub-Kahan bidiagonalization + QR) ───────────────────────────

    /**
     * Thin SVD of A(m×n), m >= n.
     * Returns [U(m×n), s(n), VT(n×n)].
     * For m < n: transpose, compute, swap U/VT.
     * Pure-PHP fallback — for large matrices, use FFI/LAPACK.
     */
    public static function svd(array $A, int $m, int $n): array
    {
        $transposed = false;
        if ($m < $n) {
            $A = self::transposeRaw($A, $m, $n);
            [$m, $n] = [$n, $m];
            $transposed = true;
        }

        // Bidiagonalization via Householder reflections
        $B  = $A;  // working copy, will become bidiagonal
        $U  = self::eye($m);
        $V  = self::eye($n);
        $d  = array_fill(0, $n, 0.0);  // diagonal
        $e  = array_fill(0, $n-1, 0.0); // super-diagonal

        for ($k = 0; $k < $n; ++$k) {
            // Left Householder to zero below B[k][k]
            $x = [];
            for ($i = $k; $i < $m; ++$i) $x[] = $B[$i][$k];
            [$v, $beta] = self::householder($x);
            if ($beta != 0.0) {
                // B = (I - beta*v*v^T) * B  (rows k..m, cols k..n)
                for ($j = $k; $j < $n; ++$j) {
                    $s = 0.0;
                    for ($i = 0; $i < count($v); ++$i) $s += $v[$i] * $B[$k+$i][$j];
                    $s *= $beta;
                    for ($i = 0; $i < count($v); ++$i) $B[$k+$i][$j] -= $s * $v[$i];
                }
                // Accumulate U
                for ($j = 0; $j < $m; ++$j) {
                    $s = 0.0;
                    for ($i = 0; $i < count($v); ++$i) $s += $v[$i] * $U[$j][$k+$i];
                    $s *= $beta;
                    for ($i = 0; $i < count($v); ++$i) $U[$j][$k+$i] -= $s * $v[$i];
                }
            }
            $d[$k] = $B[$k][$k];

            if ($k < $n-1) {
                // Right Householder to zero right of B[k][k+1]
                $x = [];
                for ($j = $k+1; $j < $n; ++$j) $x[] = $B[$k][$j];
                [$v, $beta] = self::householder($x);
                if ($beta != 0.0) {
                    for ($i = $k; $i < $m; ++$i) {
                        $s = 0.0;
                        for ($j = 0; $j < count($v); ++$j) $s += $v[$j] * $B[$i][$k+1+$j];
                        $s *= $beta;
                        for ($j = 0; $j < count($v); ++$j) $B[$i][$k+1+$j] -= $s * $v[$j];
                    }
                    for ($i = 0; $i < $n; ++$i) {
                        $s = 0.0;
                        for ($j = 0; $j < count($v); ++$j) $s += $v[$j] * $V[$i][$k+1+$j];
                        $s *= $beta;
                        for ($j = 0; $j < count($v); ++$j) $V[$i][$k+1+$j] -= $s * $v[$j];
                    }
                }
                $e[$k] = $B[$k][$k+1];
            }
        }

        // QR iteration on bidiagonal matrix to extract singular values
        $maxIter = 1000 * $n;
        for ($iter = 0; $iter < $maxIter && $n > 1; ++$iter) {
            // check convergence of last super-diagonal element
            if (abs($e[$n-2]) <= 1e-14 * (abs($d[$n-2]) + abs($d[$n-1]))) {
                --$n; continue;
            }
            // Golub-Kahan SVD step (simplified: just deflate)
            // Use Givens rotations on the bidiagonal
            $mu = self::wilkinsonShift($d[$n-2], $e[$n-2], $d[$n-1]);
            $y  = $d[0]*$d[0] - $mu;
            $z  = $d[0] * $e[0];
            for ($k = 0; $k < $n-1; ++$k) {
                [$c, $s] = self::givens($y, $z);
                // Apply right Givens G(k,k+1,theta) to B columns k and k+1
                for ($i = 0; $i <= min($k+1, count($d)-1); ++$i) {
                    if ($i === $k) {
                        [$d[$k], $e[$k]] = [$c*$d[$k] + $s*$e[$k], -$s*$d[$k] + $c*$e[$k]];
                    }
                }
                // accumulate in V
                for ($i = 0; $i < count($V); ++$i) {
                    $vk  = $V[$i][$k];
                    $vk1 = $V[$i][$k+1];
                    $V[$i][$k]   = $c*$vk  + $s*$vk1;
                    $V[$i][$k+1] = -$s*$vk + $c*$vk1;
                }
                $y = $d[$k]; $z = $s * ($k+1 < $n-1 ? $e[$k] : 0.0);
                // Left Givens
                [$c, $s] = self::givens($y, $z);
                $d[$k] = $c*$d[$k] + $s*($k < count($e) ? $e[$k] : 0.0);
                if ($k < count($e)) $e[$k] = $c*$e[$k] - $s*$d[$k];
                if ($k+1 < count($d)) {
                    $y = $e[$k] ?? 0.0; $z = $s * $d[$k+1];
                }
                for ($i = 0; $i < count($U[0]); ++$i) {
                    $uk  = $U[$k][$i];
                    $uk1 = $U[$k+1][$i];
                    $U[$k][$i]   = $c*$uk  + $s*$uk1;
                    $U[$k+1][$i] = -$s*$uk + $c*$uk1;
                }
            }
        }
        // ensure non-negative singular values
        $mm = count($d);
        for ($i = 0; $i < $mm; ++$i) {
            if ($d[$i] < 0) {
                $d[$i] = -$d[$i];
                for ($j = 0; $j < count($U); ++$j) $U[$j][$i] = -$U[$j][$i];
            }
        }
        // sort descending
        arsort($d);
        $ord = array_keys($d);
        $d   = array_values($d);
        $Uo  = $U; $Vo = $V;
        foreach ($ord as $newIdx => $oldIdx) {
            for ($i = 0; $i < count($U); ++$i) $Uo[$i][$newIdx] = $U[$i][$oldIdx];
            for ($i = 0; $i < count($V); ++$i) $Vo[$i][$newIdx] = $V[$i][$oldIdx];
        }
        $VT = self::transposeRaw($Vo, count($Vo), count($Vo[0]));
        if ($transposed) return [$Vo, $d, self::transposeRaw($Uo, count($Uo), count($Uo[0]))];
        return [$Uo, $d, $VT];
    }

    private static function householder(array $x): array
    {
        $n    = count($x);
        $norm = 0.0;
        foreach ($x as $v) $norm += $v*$v;
        $norm = sqrt($norm);
        if ($norm == 0.0) return [$x, 0.0];
        $v    = $x;
        $v[0] += ($x[0] >= 0 ? 1 : -1) * $norm;
        $vNorm = 0.0;
        foreach ($v as $vi) $vNorm += $vi*$vi;
        $beta = 2.0 / $vNorm;
        return [$v, $beta];
    }

    private static function wilkinsonShift(float $a, float $b, float $c): float
    {
        $d   = ($a*$a + $c*$c - $b*$b - $a*$a) / 2.0;  // simplified
        $mu  = $c*$c + $d - ($d >= 0 ? 1 : -1) * sqrt($d*$d + $b*$b*$c*$c / max(1e-300, $c*$c));
        return $mu;
    }

    private static function givens(float $a, float $b): array
    {
        if ($b == 0.0) return [1.0, 0.0];
        if (abs($b) > abs($a)) { $t = -$a/$b; $s = 1/sqrt(1+$t*$t); return [$s*$t, $s]; }
        $t = -$b/$a; $c = 1/sqrt(1+$t*$t); return [$c, $c*$t];
    }

    /** Identity matrix n×n as array-of-arrays */
    public static function eye(int $n): array
    {
        $I = [];
        for ($i = 0; $i < $n; ++$i) {
            $row = array_fill(0, $n, 0.0);
            $row[$i] = 1.0;
            $I[$i] = $row;
        }
        return $I;
    }

    // ── Symmetric eigendecomposition (Jacobi iteration) ───────────────────

    /**
     * Jacobi eigendecomposition for real symmetric matrices.
     * Returns [eigenvalues (sorted ascending), eigenvectors (columns)].
     */
    public static function eigSymmetric(array $A, int $n): array
    {
        $V  = self::eye($n);
        $D  = $A;  // will become diagonal

        for ($sweep = 0; $sweep < 100 * $n * $n; ++$sweep) {
            // find largest off-diagonal element
            $maxVal = 0.0; $p = 0; $q = 1;
            for ($i = 0; $i < $n; ++$i)
                for ($j = $i+1; $j < $n; ++$j)
                    if (($v = abs($D[$i][$j])) > $maxVal) { $maxVal = $v; $p = $i; $q = $j; }
            if ($maxVal < 1e-14) break;

            // Jacobi rotation
            $theta = ($D[$q][$q] - $D[$p][$p]) / (2.0 * $D[$p][$q]);
            $t     = ($theta >= 0 ? 1 : -1) / (abs($theta) + sqrt(1 + $theta*$theta));
            $c     = 1.0 / sqrt(1 + $t*$t);
            $s     = $t * $c;

            // rotate D
            $Dpp = $D[$p][$p]; $Dqq = $D[$q][$q]; $Dpq = $D[$p][$q];
            $D[$p][$p] = $Dpp - $t*$Dpq;
            $D[$q][$q] = $Dqq + $t*$Dpq;
            $D[$p][$q] = $D[$q][$p] = 0.0;
            for ($r = 0; $r < $n; ++$r) {
                if ($r !== $p && $r !== $q) {
                    $Drp = $D[$r][$p]; $Drq = $D[$r][$q];
                    $D[$r][$p] = $D[$p][$r] = $c*$Drp - $s*$Drq;
                    $D[$r][$q] = $D[$q][$r] = $s*$Drp + $c*$Drq;
                }
                // accumulate eigenvectors
                $Vrp = $V[$r][$p]; $Vrq = $V[$r][$q];
                $V[$r][$p] = $c*$Vrp - $s*$Vrq;
                $V[$r][$q] = $s*$Vrp + $c*$Vrq;
            }
        }
        // extract eigenvalues and sort ascending
        $w = [];
        for ($i = 0; $i < $n; ++$i) $w[$i] = $D[$i][$i];
        asort($w);
        $ord = array_keys($w);
        $w   = array_values($w);
        $Vo  = $V;
        foreach ($ord as $newIdx => $oldIdx)
            for ($i = 0; $i < $n; ++$i)
                $Vo[$i][$newIdx] = $V[$i][$oldIdx];
        return [$w, $Vo];
    }

    // ── Row Echelon Form ──────────────────────────────────────────────────

    public static function ref(array $A, int $m, int $n): array
    {
        $R     = $A;
        $swaps = 0;
        $row   = 0;
        for ($col = 0; $col < $n && $row < $m; ++$col) {
            // find pivot
            $maxVal = abs($R[$row][$col]); $maxRow = $row;
            for ($i = $row+1; $i < $m; ++$i)
                if (($v = abs($R[$i][$col])) > $maxVal) { $maxVal = $v; $maxRow = $i; }
            if ($maxVal < 1e-14) continue;
            if ($maxRow !== $row) { [$R[$row], $R[$maxRow]] = [$R[$maxRow], $R[$row]]; ++$swaps; }
            $inv = 1.0 / $R[$row][$col];
            for ($i = $row+1; $i < $m; ++$i) {
                $f = $R[$i][$col] * $inv;
                for ($j = $col; $j < $n; ++$j)
                    $R[$i][$j] -= $f * $R[$row][$j];
                $R[$i][$col] = 0.0;
            }
            ++$row;
        }
        return [$R, $swaps];
    }

    public static function rref(array $A, int $m, int $n): array
    {
        [$R, ] = self::ref($A, $m, $n);
        $lead = 0;
        for ($r = 0; $r < $m; ++$r) {
            if ($lead >= $n) break;
            $i = $r;
            while ($i < $m && abs($R[$i][$lead]) < 1e-14) {
                ++$i;
                if ($i === $m) { $i = $r; ++$lead; if ($lead === $n) goto done; }
            }
            [$R[$i], $R[$r]] = [$R[$r], $R[$i]];
            $lv = $R[$r][$lead];
            if (abs($lv) > 1e-14)
                for ($j = 0; $j < $n; ++$j) $R[$r][$j] /= $lv;
            for ($i = 0; $i < $m; ++$i) {
                if ($i !== $r) {
                    $f = $R[$i][$lead];
                    for ($j = 0; $j < $n; ++$j) $R[$i][$j] -= $f * $R[$r][$j];
                }
            }
            ++$lead;
        }
        done:
        return $R;
    }

    /** Matrix rank via REF */
    public static function rank(array $A, int $m, int $n): int
    {
        [$R, ] = self::ref($A, $m, $n);
        $rank = 0;
        for ($i = 0; $i < $m; ++$i) {
            $nonzero = false;
            for ($j = 0; $j < $n; ++$j) if (abs($R[$i][$j]) > 1e-10) { $nonzero = true; break; }
            if ($nonzero) ++$rank;
        }
        return $rank;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  MATRIX CLASS
// ═══════════════════════════════════════════════════════════════════════════

class Matrix implements Tensor
{
    /** @var array<int,array<int,float>> Row-major array-of-arrays */
    private array $a;
    private int   $m;  // rows
    private int   $n;  // cols

    private function __construct(array $a, int $m, int $n)
    {
        $this->a = $a; $this->m = $m; $this->n = $n;
    }

    // ── factories ────────────────────────────────────────────────────────

    /** Build from array-of-arrays with validation */
    public static function build(array $a): self
    {
        $m = count($a);
        if ($m === 0) throw new \InvalidArgumentException("Matrix must have at least 1 row");
        $n = count(reset($a));
        foreach ($a as $i => $row) {
            if (count($row) !== $n)
                throw new \InvalidArgumentException("Row $i has ".count($row)." elements, expected $n");
        }
        $out = [];
        foreach ($a as $i => $row) { $r = []; foreach ($row as $v) $r[] = (float)$v; $out[$i] = $r; }
        return new self($out, $m, $n);
    }

    /** Build without validation (trusted input) */
    public static function quick(array $a): self
    {
        $m = count($a); $n = $m > 0 ? count($a[0]) : 0;
        return new self($a, $m, $n);
    }

    public static function zeros(int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) $a[$i] = array_fill(0, $n, 0.0);
        return new self($a, $m, $n);
    }

    public static function ones(int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) $a[$i] = array_fill(0, $n, 1.0);
        return new self($a, $m, $n);
    }

    public static function fill(float $val, int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) $a[$i] = array_fill(0, $n, $val);
        return new self($a, $m, $n);
    }

    public static function eye(int $n): self
    {
        return new self(LinAlg::eye($n), $n, $n);
    }

    public static function identity(int $n): self { return self::eye($n); }

    public static function diagonal(array $elements): self
    {
        $n = count($elements);
        $a = [];
        for ($i = 0; $i < $n; ++$i) {
            $row = array_fill(0, $n, 0.0); $row[$i] = (float)$elements[$i]; $a[$i] = $row;
        }
        return new self($a, $n, $n);
    }

    public static function rand(int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) $row[$j] = mt_rand() / mt_getrandmax();
            $a[$i] = $row;
        }
        return new self($a, $m, $n);
    }

    public static function gaussian(int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) {
                // Box-Muller
                $u = (mt_rand() + 1) / (mt_getrandmax() + 1);
                $v = (mt_rand() + 1) / (mt_getrandmax() + 1);
                $row[$j] = sqrt(-2*log($u)) * cos(2*M_PI*$v);
            }
            $a[$i] = $row;
        }
        return new self($a, $m, $n);
    }

    public static function uniform(int $m, int $n, float $lo = -1.0, float $hi = 1.0): self
    {
        $a = [];
        $range = $hi - $lo;
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) $row[$j] = $lo + (mt_rand() / mt_getrandmax()) * $range;
            $a[$i] = $row;
        }
        return new self($a, $m, $n);
    }

    public static function poisson(int $m, int $n, float $lambda = 1.0): self
    {
        $a = []; $L = exp(-$lambda);
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) {
                $k = 0; $p = 1.0;
                do { ++$k; $p *= mt_rand()/mt_getrandmax(); } while ($p > $L);
                $row[$j] = (float)($k - 1);
            }
            $a[$i] = $row;
        }
        return new self($a, $m, $n);
    }

    // ── Tensor interface ──────────────────────────────────────────────────

    public function shape(): array    { return [$this->m, $this->n]; }
    public function shapeString(): string { return "{$this->m} x {$this->n}"; }
    public function isSquare(): bool  { return $this->m === $this->n; }
    public function size(): int       { return $this->m * $this->n; }
    public function ndim(): int       { return 2; }
    public function dtype(): int      { return Tensor::FLOAT; }
    public function m(): int          { return $this->m; }
    public function n(): int          { return $this->n; }
    public function count(): int      { return $this->m; }  // Countable: rows

    public function asArray(): array  { return $this->a; }
    public function jsonSerialize(): array { return $this->a; }
    public function __toString(): string
    {
        $rows = [];
        foreach ($this->a as $row) $rows[] = '[ '.implode(', ', array_map(fn($v)=>round($v,4), $row)).' ]';
        return implode("\n", $rows);
    }

    public function getIterator(): \ArrayIterator
    {
        return new \ArrayIterator($this->a);
    }

    // ── row/column accessors ──────────────────────────────────────────────

    public function rowAsVector(int $index): Vector
    {
        if ($index < 0 || $index >= $this->m)
            throw new \OutOfBoundsException("Row $index out of bounds");
        return Vector::quick($this->a[$index]);
    }

    public function columnAsVector(int $index): ColumnVector
    {
        if ($index < 0 || $index >= $this->n)
            throw new \OutOfBoundsException("Column $index out of bounds");
        $col = [];
        for ($i = 0; $i < $this->m; ++$i) $col[$i] = $this->a[$i][$index];
        return ColumnVector::quick($col);
    }

    public function diagonal(): Vector
    {
        $k = min($this->m, $this->n); $d = [];
        for ($i = 0; $i < $k; ++$i) $d[$i] = $this->a[$i][$i];
        return Vector::quick($d);
    }

    public function flatten(): Vector
    {
        $flat = [];
        foreach ($this->a as $row) foreach ($row as $v) $flat[] = $v;
        return Vector::quick($flat);
    }

    // ── reductions ────────────────────────────────────────────────────────

    public function sum(): float
    {
        $s = 0.0;
        foreach ($this->a as $row) foreach ($row as $v) $s += $v;
        return $s;
    }

    public function product(): float
    {
        $p = 1.0;
        foreach ($this->a as $row) foreach ($row as $v) $p *= $v;
        return $p;
    }

    public function min(): float
    {
        $m = INF;
        foreach ($this->a as $row) foreach ($row as $v) if ($v < $m) $m = $v;
        return $m;
    }

    public function max(): float
    {
        $m = -INF;
        foreach ($this->a as $row) foreach ($row as $v) if ($v > $m) $m = $v;
        return $m;
    }

    public function mean(): float { return $this->sum() / $this->size(); }

    public function median(): float
    {
        $flat = [];
        foreach ($this->a as $row) foreach ($row as $v) $flat[] = $v;
        sort($flat); $sz = count($flat);
        return $sz % 2 === 0 ? ($flat[$sz/2-1] + $flat[$sz/2]) / 2.0 : $flat[($sz-1)/2];
    }

    public function variance(?Tensor $mean = null): float
    {
        $mu = $mean instanceof self ? $mean->mean() : $this->mean();
        $s  = 0.0; $sz = $this->size();
        foreach ($this->a as $row) foreach ($row as $v) $s += ($v - $mu) ** 2;
        return $s / $sz;
    }

    public function trace(): float
    {
        if (!$this->isSquare()) throw new \RuntimeException("Trace requires square matrix");
        $t = 0.0; for ($i = 0; $i < $this->m; ++$i) $t += $this->a[$i][$i]; return $t;
    }

    public function rowSums(): ColumnVector
    {
        $s = [];
        foreach ($this->a as $i => $row) { $t = 0.0; foreach ($row as $v) $t += $v; $s[$i] = $t; }
        return ColumnVector::quick($s);
    }

    public function columnSums(): Vector
    {
        $s = array_fill(0, $this->n, 0.0);
        foreach ($this->a as $row) foreach ($row as $j => $v) $s[$j] += $v;
        return Vector::quick($s);
    }

    public function rowMeans(): ColumnVector
    {
        return $this->rowSums()->divide($this->n);
    }

    public function columnMeans(): Vector
    {
        return $this->columnSums()->divide($this->m);
    }

    public function rowMaxima(): ColumnVector
    {
        $s = [];
        foreach ($this->a as $i => $row) $s[$i] = max($row);
        return ColumnVector::quick($s);
    }

    public function columnMaxima(): Vector
    {
        $s = array_fill(0, $this->n, -INF);
        foreach ($this->a as $row)
            foreach ($row as $j => $v) if ($v > $s[$j]) $s[$j] = $v;
        return Vector::quick($s);
    }

    public function rowMinima(): ColumnVector
    {
        $s = [];
        foreach ($this->a as $i => $row) $s[$i] = min($row);
        return ColumnVector::quick($s);
    }

    public function columnMinima(): Vector
    {
        $s = array_fill(0, $this->n, INF);
        foreach ($this->a as $row)
            foreach ($row as $j => $v) if ($v < $s[$j]) $s[$j] = $v;
        return Vector::quick($s);
    }

    public function rowNorms(float $p = 2.0): ColumnVector
    {
        $s = [];
        foreach ($this->a as $i => $row) {
            if ($p == 2.0) { $t=0.0; foreach($row as $v) $t+=$v*$v; $s[$i]=sqrt($t); }
            elseif ($p == 1.0) { $t=0.0; foreach($row as $v) $t+=abs($v); $s[$i]=$t; }
            elseif (is_infinite($p)) { $t=0.0; foreach($row as $v) $t=max($t,abs($v)); $s[$i]=$t; }
            else { $t=0.0; foreach($row as $v) $t+=pow(abs($v),$p); $s[$i]=pow($t,1/$p); }
        }
        return ColumnVector::quick($s);
    }

    public function columnNorms(float $p = 2.0): Vector
    {
        $s = array_fill(0, $this->n, 0.0);
        if ($p == 2.0) {
            foreach ($this->a as $row) foreach ($row as $j => $v) $s[$j] += $v*$v;
            for ($j = 0; $j < $this->n; ++$j) $s[$j] = sqrt($s[$j]);
        } elseif ($p == 1.0) {
            foreach ($this->a as $row) foreach ($row as $j => $v) $s[$j] += abs($v);
        } else {
            foreach ($this->a as $row) foreach ($row as $j => $v) $s[$j] += pow(abs($v), $p);
            for ($j = 0; $j < $this->n; ++$j) $s[$j] = pow($s[$j], 1/$p);
        }
        return Vector::quick($s);
    }

    // ── element-wise unary ops ────────────────────────────────────────────

    private function mapElements(callable $fn): self
    {
        $a = [];
        foreach ($this->a as $i => $row) {
            $r = [];
            foreach ($row as $v) $r[] = $fn($v);
            $a[$i] = $r;
        }
        return new self($a, $this->m, $this->n);
    }

    public function round(int $precision = 0): self { return $this->mapElements(fn($v)=>round($v,$precision)); }
    public function floor(): self   { return $this->mapElements('floor'); }
    public function ceil(): self    { return $this->mapElements('ceil'); }
    public function abs(): self     { return $this->mapElements('abs'); }
    public function square(): self  { return $this->mapElements(fn($v)=>$v*$v); }
    public function sqrt(): self    { return $this->mapElements('sqrt'); }
    public function exp(): self     { return $this->mapElements('exp'); }
    public function expm1(): self   { return $this->mapElements('expm1'); }
    public function log(float $base = M_E): self
    {
        return $base === M_E ? $this->mapElements('log') : $this->mapElements(fn($v)=>log($v,$base));
    }
    public function log1p(): self   { return $this->mapElements('log1p'); }
    public function sin(): self     { return $this->mapElements('sin'); }
    public function cos(): self     { return $this->mapElements('cos'); }
    public function tan(): self     { return $this->mapElements('tan'); }
    public function asin(): self    { return $this->mapElements('asin'); }
    public function acos(): self    { return $this->mapElements('acos'); }
    public function atan(): self    { return $this->mapElements('atan'); }
    public function sinh(): self    { return $this->mapElements('sinh'); }
    public function cosh(): self    { return $this->mapElements('cosh'); }
    public function tanh(): self    { return $this->mapElements('tanh'); }
    public function negate(): self  { return $this->mapElements(fn($v)=>-$v); }
    public function reciprocal(): self { return $this->mapElements(fn($v)=>1.0/$v); }
    public function map(callable $fn): self { return $this->mapElements($fn); }

    public function clip(float $min, float $max): self
    { return $this->mapElements(fn($v)=>max($min, min($max, $v))); }
    public function clipLower(float $min): self
    { return $this->mapElements(fn($v)=>max($min, $v)); }
    public function clipUpper(float $max): self
    { return $this->mapElements(fn($v)=>min($max, $v)); }

    public function reduce(callable $fn, int|float $initial = 0): int|float
    {
        $acc = $initial;
        foreach ($this->a as $row) foreach ($row as $v) $acc = $fn($acc, $v);
        return $acc;
    }

    // ── element-wise arithmetic ───────────────────────────────────────────

    public function add(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v+(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a = []; foreach ($this->a as $i=>$row) { $r=[]; foreach($row as $j=>$v) $r[]=$v+$b->a[$i][$j]; $a[$i]=$r; }
            return new self($a, $this->m, $this->n);
        }
        if ($b instanceof Vector) { // broadcast row vector
            if ($b->n() !== $this->n) throw new \RuntimeException("Shape mismatch");
            $bArr = $b->asArray();
            $a = [];
            foreach ($this->a as $i=>$row) { $r=[]; foreach($row as $j=>$v) $r[]=$v+$bArr[$j]; $a[$i]=$r; }
            return new self($a, $this->m, $this->n);
        }
        if ($b instanceof ColumnVector) {
            if ($b->n() !== $this->m) throw new \RuntimeException("Shape mismatch");
            $bArr = $b->asArray();
            $a = [];
            foreach ($this->a as $i=>$row) { $r=[]; foreach($row as $j=>$v) $r[]=$v+$bArr[$i]; $a[$i]=$r; }
            return new self($a, $this->m, $this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function subtract(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v-(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v-$b->a[$i][$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        if ($b instanceof Vector) {
            if ($b->n()!==$this->n) throw new \RuntimeException("Shape mismatch");
            $bArr=$b->asArray(); $a=[];
            foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v-$bArr[$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        if ($b instanceof ColumnVector) {
            if ($b->n()!==$this->m) throw new \RuntimeException("Shape mismatch");
            $bArr=$b->asArray(); $a=[];
            foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v-$bArr[$i];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function multiply(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v*(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v*$b->a[$i][$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        if ($b instanceof Vector) {
            if ($b->n()!==$this->n) throw new \RuntimeException("Shape mismatch");
            $bArr=$b->asArray(); $a=[];
            foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v*$bArr[$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        if ($b instanceof ColumnVector) {
            if ($b->n()!==$this->m) throw new \RuntimeException("Shape mismatch");
            $bArr=$b->asArray(); $a=[];
            foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v*$bArr[$i];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function divide(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v/(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v/$b->a[$i][$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function pow(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v**(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=pow($v,$b->a[$i][$j]);$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function mod(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>fmod($v,(float)$b));
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=fmod($v,$b->a[$i][$j]);$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    // ── comparisons ───────────────────────────────────────────────────────

    private function cmp(int|float|Tensor $b, callable $fn): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$fn($v,(float)$b)?1.0:0.0);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$fn($v,$b->a[$i][$j])?1.0:0.0;$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function equal(int|float|Tensor $b): self      { return $this->cmp($b, fn($x,$y)=>$x==$y); }
    public function notEqual(int|float|Tensor $b): self   { return $this->cmp($b, fn($x,$y)=>$x!=$y); }
    public function greater(int|float|Tensor $b): self    { return $this->cmp($b, fn($x,$y)=>$x>$y); }
    public function greaterEqual(int|float|Tensor $b): self { return $this->cmp($b, fn($x,$y)=>$x>=$y); }
    public function less(int|float|Tensor $b): self       { return $this->cmp($b, fn($x,$y)=>$x<$y); }
    public function lessEqual(int|float|Tensor $b): self  { return $this->cmp($b, fn($x,$y)=>$x<=$y); }

    // ── structure ops ─────────────────────────────────────────────────────

    public function transpose(): self
    {
        return new self(LinAlg::transposeRaw($this->a, $this->m, $this->n), $this->n, $this->m);
    }

    public function asType(int $dtype): self
    {
        $fn = $dtype === Tensor::INT ? fn($v)=>(float)(int)$v : fn($v)=>(float)$v;
        return $this->mapElements($fn);
    }

    public function stack(self $b): self
    {
        if ($b->n !== $this->n) throw new \RuntimeException("Column count mismatch");
        return new self(array_merge($this->a, $b->a), $this->m + $b->m, $this->n);
    }

    public function augment(self $b): self
    {
        if ($b->m !== $this->m) throw new \RuntimeException("Row count mismatch");
        $a = [];
        foreach ($this->a as $i => $row) $a[$i] = array_merge($row, $b->a[$i]);
        return new self($a, $this->m, $this->n + $b->n);
    }

    public function repeat(int $m, int $n): self
    {
        $a = [];
        for ($ri = 0; $ri < $this->m * $m; ++$ri) {
            $srcRow = $this->a[$ri % $this->m];
            $row    = [];
            for ($rj = 0; $rj < $n; ++$rj) $row = array_merge($row, $srcRow);
            $a[$ri] = $row;
        }
        return new self($a, $this->m * $m, $this->n * $n);
    }

    public function rowExclude(int $index): self
    {
        $a = $this->a; unset($a[$index]); $a = array_values($a);
        return new self($a, $this->m - 1, $this->n);
    }

    public function columnExclude(int $index): self
    {
        $a = [];
        foreach ($this->a as $i => $row) { unset($row[$index]); $a[$i] = array_values($row); }
        return new self($a, $this->m, $this->n - 1);
    }

    public function softmax(): self
    {
        $a = [];
        foreach ($this->a as $i => $row) {
            $max = max($row);
            $exp = array_map(fn($v)=>exp($v-$max), $row);
            $sum = array_sum($exp);
            $a[$i] = array_map(fn($v)=>$v/$sum, $exp);
        }
        return new self($a, $this->m, $this->n);
    }

    public function symmetric(): bool
    {
        if (!$this->isSquare()) return false;
        for ($i = 0; $i < $this->m; ++$i)
            for ($j = 0; $j < $i; ++$j)
                if (abs($this->a[$i][$j] - $this->a[$j][$i]) > 1e-10) return false;
        return true;
    }

    public function positiveDefinite(): bool
    {
        if (!$this->symmetric()) return false;
        try { $this->cholesky(); return true; } catch (\RuntimeException) { return false; }
    }

    public function positiveHalfDefinite(): bool
    {
        if (!$this->symmetric()) return false;
        try {
            [$w, ] = $this->eig(true);
            foreach ($w->asArray() as $v) if ($v < -1e-10) return false;
            return true;
        } catch (\RuntimeException) { return false; }
    }

    // ── linear algebra ────────────────────────────────────────────────────

    /** Matrix–matrix multiply: returns Matrix (m×p) */
    public function matmul(self $b): self
    {
        if ($this->n !== $b->m) throw new \RuntimeException("Incompatible dimensions for matmul");
        $blas = Blas::getInstance();
        if ($blas->available()) {
            $res = $blas->dgemm($this->a, $this->m, $this->n, $b->a, $b->n);
            return new self($res, $this->m, $b->n);
        }
        return new self(LinAlg::matmul($this->a, $b->a, $this->m, $this->n, $b->n), $this->m, $b->n);
    }

    /** Matrix–vector: A·x → ColumnVector */
    public function dot(Vector $x): ColumnVector
    {
        if ($this->n !== $x->n()) throw new \RuntimeException("Incompatible dimensions for dot");
        $blas = Blas::getInstance();
        if ($blas->available()) {
            $y = $blas->dgemv($this->a, $this->m, $this->n, $x->asArray());
            return ColumnVector::quick($y);
        }
        return ColumnVector::quick(LinAlg::matvec($this->a, $x->asArray(), $this->m, $this->n));
    }

    public function inverse(): self
    {
        if (!$this->isSquare()) throw new \RuntimeException("Matrix must be square");
        $blas = Blas::getInstance();
        if ($blas->available() && $blas->hasLapack()) {
            return new self($blas->inverse($this->a, $this->m), $this->m, $this->m);
        }
        return new self(LinAlg::inverse($this->a, $this->m), $this->m, $this->m);
    }

    public function det(): float
    {
        if (!$this->isSquare()) throw new \RuntimeException("Matrix must be square");
        return LinAlg::det($this->a, $this->m);
    }

    public function rank(): int { return LinAlg::rank($this->a, $this->m, $this->n); }

    public function solve(self $b): self
    {
        if (!$this->isSquare()) throw new \RuntimeException("Coefficient matrix must be square");
        if ($b->m !== $this->m) throw new \RuntimeException("RHS row count mismatch");
        $blas = Blas::getInstance();
        if ($blas->available() && $blas->hasLapack()) {
            $res = $blas->solve($this->a, $b->a, $this->m, $b->n);
            return new self($res, $this->m, $b->n);
        }
        return new self(LinAlg::solve($this->a, $b->a, $this->m, $b->n), $this->m, $b->n);
    }

    /** LU decomposition: returns ['L'=>Matrix, 'U'=>Matrix, 'P'=>Matrix] */
    public function lu(): array
    {
        if (!$this->isSquare()) throw new \RuntimeException("LU requires square matrix");
        $n = $this->m;
        [$LU, $ipiv] = LinAlg::lu($this->a, $n);
        // Separate L and U
        $L = LinAlg::eye($n); $U = [];
        for ($i = 0; $i < $n; ++$i) $U[$i] = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; ++$i) {
            for ($j = 0; $j < $n; ++$j) {
                if ($j < $i)       $L[$i][$j] = $LU[$i][$j];
                elseif ($j === $i) { $L[$i][$j] = 1.0; $U[$i][$j] = $LU[$i][$j]; }
                else               $U[$i][$j] = $LU[$i][$j];
            }
        }
        // Build permutation matrix from ipiv
        $pArr = LinAlg::eye($n);
        for ($i = 0; $i < $n; ++$i) if ($ipiv[$i] !== $i) [$pArr[$i], $pArr[$ipiv[$i]]] = [$pArr[$ipiv[$i]], $pArr[$i]];
        return ['L' => new self($L,$n,$n), 'U' => new self($U,$n,$n), 'P' => new self($pArr,$n,$n)];
    }

    public function cholesky(): self
    {
        if (!$this->isSquare()) throw new \RuntimeException("Cholesky requires square matrix");
        $blas = Blas::getInstance();
        if ($blas->available() && $blas->hasLapack()) {
            return new self($blas->dpotrf($this->a, $this->m), $this->m, $this->m);
        }
        return new self(LinAlg::cholesky($this->a, $this->m), $this->m, $this->m);
    }

    /** SVD: returns ['U'=>Matrix, 's'=>Vector, 'VT'=>Matrix] */
    public function svd(): array
    {
        $blas = Blas::getInstance();
        if ($blas->available() && $blas->hasLapack()) {
            [$U, $s, $VT] = $blas->dgesvd($this->a, $this->m, $this->n);
            return ['U' => new self($U, $this->m, $this->m),
                    's' => Vector::quick($s),
                    'VT'=> new self($VT, $this->n, $this->n)];
        }
        [$U, $s, $VT] = LinAlg::svd($this->a, $this->m, $this->n);
        return ['U' => new self($U, $this->m, count($U[0])),
                's' => Vector::quick($s),
                'VT'=> new self($VT, count($VT), $this->n)];
    }

    /**
     * Eigendecomposition: returns ['values'=>Vector, 'vectors'=>Matrix]
     * symmetric=true uses symmetric algorithm (real eigenvalues guaranteed).
     */
    public function eig(bool $symmetric = false): array
    {
        if (!$this->isSquare()) throw new \RuntimeException("Eig requires square matrix");
        $blas = Blas::getInstance();
        if ($symmetric) {
            if ($blas->available() && $blas->hasLapack()) {
                [$w, $v] = $blas->dsyev($this->a, $this->m);
                return ['values' => Vector::quick($w), 'vectors' => new self($v, $this->m, $this->m)];
            }
            [$w, $v] = LinAlg::eigSymmetric($this->a, $this->m);
            return ['values' => Vector::quick($w), 'vectors' => new self($v, $this->m, $this->m)];
        }
        if ($blas->available() && $blas->hasLapack()) {
            [$wr, $wi, $vr] = $blas->dgeev($this->a, $this->m);
            return ['values' => Vector::quick($wr), 'values_imag' => Vector::quick($wi),
                    'vectors' => new self($vr, $this->m, $this->m)];
        }
        // Pure PHP general eig: use symmetric approximation or throw
        throw new \RuntimeException("General eigendecomposition requires LAPACK (set USE_FFI=true and install libopenblas)");
    }

    public function ref(): array
    {
        [$R, $swaps] = LinAlg::ref($this->a, $this->m, $this->n);
        return [new self($R, $this->m, $this->n), $swaps];
    }

    public function rref(): self
    {
        return new self(LinAlg::rref($this->a, $this->m, $this->n), $this->m, $this->n);
    }

    public function pseudoinverse(): self
    {
        ['U'=>$U, 's'=>$s, 'VT'=>$VT] = $this->svd();
        $sArr = $s->asArray(); $tol = 1e-10 * max($this->m,$this->n) * max($sArr);
        $sInv = array_map(fn($v)=>abs($v)>$tol?1.0/$v:0.0, $sArr);
        $SInv = self::diagonal($sInv);
        // A+ = V * S+ * U^T
        return $VT->transpose()->matmul($SInv)->matmul($U->transpose());
    }

    public function convolve(self $kernel, int $stride = 1): self
    {
        $km = $kernel->m; $kn = $kernel->n;
        $om = (int)(($this->m - $km) / $stride) + 1;
        $on = (int)(($this->n - $kn) / $stride) + 1;
        $a  = [];
        for ($i = 0; $i < $om; ++$i) {
            $row = [];
            for ($j = 0; $j < $on; ++$j) {
                $s = 0.0;
                for ($ki = 0; $ki < $km; ++$ki)
                    for ($kj = 0; $kj < $kn; ++$kj)
                        $s += $this->a[$i*$stride+$ki][$j*$stride+$kj] * $kernel->a[$ki][$kj];
                $row[$j] = $s;
            }
            $a[$i] = $row;
        }
        return new self($a, $om, $on);
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private function checkShape(self $b): void
    {
        if ($b->m !== $this->m || $b->n !== $this->n)
            throw new \RuntimeException("Shape mismatch: {$this->m}×{$this->n} vs {$b->m}×{$b->n}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  VECTOR CLASS
// ═══════════════════════════════════════════════════════════════════════════

class Vector implements Tensor
{
    protected array $a;
    protected int   $n;

    protected function __construct(array $a, int $n)
    {
        $this->a = $a; $this->n = $n;
    }

    // ── factories ────────────────────────────────────────────────────────

    public static function build(array $a): static
    {
        $flat = []; foreach ($a as $v) $flat[] = (float)$v;
        return new static($flat, count($flat));
    }

    public static function quick(array $a): static
    {
        return new static(array_values($a), count($a));
    }

    public static function zeros(int $n): static
    { return new static(array_fill(0, $n, 0.0), $n); }

    public static function ones(int $n): static
    { return new static(array_fill(0, $n, 1.0), $n); }

    public static function fill(float $val, int $n): static
    { return new static(array_fill(0, $n, $val), $n); }

    public static function rand(int $n): static
    {
        $a = []; for ($i=0;$i<$n;++$i) $a[]= mt_rand()/mt_getrandmax();
        return new static($a, $n);
    }

    public static function gaussian(int $n): static
    {
        $a = [];
        for ($i=0;$i<$n;++$i) {
            $u=(mt_rand()+1)/(mt_getrandmax()+1); $v=(mt_rand()+1)/(mt_getrandmax()+1);
            $a[]=sqrt(-2*log($u))*cos(2*M_PI*$v);
        }
        return new static($a, $n);
    }

    public static function uniform(int $n, float $lo=-1.0, float $hi=1.0): static
    {
        $a=[]; $r=$hi-$lo;
        for($i=0;$i<$n;++$i) $a[]=$lo+(mt_rand()/mt_getrandmax())*$r;
        return new static($a,$n);
    }

    // ── Tensor interface ──────────────────────────────────────────────────

    public function shape(): array   { return [$this->n]; }
    public function shapeString(): string { return (string)$this->n; }
    public function isSquare(): bool { return false; }
    public function size(): int      { return $this->n; }
    public function ndim(): int      { return 1; }
    public function dtype(): int     { return Tensor::FLOAT; }
    public function n(): int         { return $this->n; }
    public function count(): int     { return $this->n; }
    public function asArray(): array { return $this->a; }
    public function jsonSerialize(): array { return $this->a; }
    public function __toString(): string
    { return '[ '.implode(', ', array_map(fn($v)=>round($v,4), $this->a)).' ]'; }

    public function getIterator(): \ArrayIterator
    { return new \ArrayIterator($this->a); }

    // ── conversions ───────────────────────────────────────────────────────

    public function asColumnVector(): ColumnVector { return ColumnVector::quick($this->a); }
    public function asRowMatrix(): Matrix { return Matrix::quick([$this->a]); }
    public function asColumnMatrix(): Matrix
    { $a=[]; foreach($this->a as $v) $a[]= [$v]; return Matrix::quick($a); }
    public function flatten(): Vector { return Vector::quick($this->a); }

    // ── reductions ────────────────────────────────────────────────────────

    public function sum(): float   { return array_sum($this->a); }
    public function product(): float
    { $p=1.0; foreach($this->a as $v)$p*=$v; return $p; }
    public function min(): float   { return min($this->a); }
    public function max(): float   { return max($this->a); }
    public function mean(): float  { return array_sum($this->a) / $this->n; }

    public function median(): float
    {
        $s = $this->a; sort($s);
        return $this->n%2===0 ? ($s[$this->n/2-1]+$s[$this->n/2])/2.0 : $s[($this->n-1)/2];
    }

    public function variance(?Tensor $mean=null): float
    {
        $mu = $mean ? $mean->mean() : $this->mean();
        $s=0.0; foreach($this->a as $v) $s+=($v-$mu)**2;
        return $s/$this->n;
    }

    public function argmin(): int
    {
        $min=INF; $idx=0;
        foreach($this->a as $i=>$v) if($v<$min){$min=$v;$idx=$i;}
        return $idx;
    }

    public function argmax(): int
    {
        $max=-INF; $idx=0;
        foreach($this->a as $i=>$v) if($v>$max){$max=$v;$idx=$i;}
        return $idx;
    }

    // ── element-wise unary ops ────────────────────────────────────────────

    private function mapEl(callable $fn): static
    {
        $a=[]; foreach($this->a as $v) $a[]=$fn($v);
        return new static($a,$this->n);
    }

    public function round(int $precision=0): static { return $this->mapEl(fn($v)=>round($v,$precision)); }
    public function floor(): static  { return $this->mapEl('floor'); }
    public function ceil(): static   { return $this->mapEl('ceil'); }
    public function abs(): static    { return $this->mapEl('abs'); }
    public function square(): static { return $this->mapEl(fn($v)=>$v*$v); }
    public function sqrt(): static   { return $this->mapEl('sqrt'); }
    public function exp(): static    { return $this->mapEl('exp'); }
    public function expm1(): static  { return $this->mapEl('expm1'); }
    public function log(float $base=M_E): static
    { return $base===M_E?$this->mapEl('log'):$this->mapEl(fn($v)=>log($v,$base)); }
    public function log1p(): static  { return $this->mapEl('log1p'); }
    public function sin(): static    { return $this->mapEl('sin'); }
    public function cos(): static    { return $this->mapEl('cos'); }
    public function tan(): static    { return $this->mapEl('tan'); }
    public function asin(): static   { return $this->mapEl('asin'); }
    public function acos(): static   { return $this->mapEl('acos'); }
    public function atan(): static   { return $this->mapEl('atan'); }
    public function sinh(): static   { return $this->mapEl('sinh'); }
    public function cosh(): static   { return $this->mapEl('cosh'); }
    public function tanh(): static   { return $this->mapEl('tanh'); }
    public function negate(): static { return $this->mapEl(fn($v)=>-$v); }
    public function reciprocal(): static { return $this->mapEl(fn($v)=>1.0/$v); }
    public function map(callable $fn): static { return $this->mapEl($fn); }

    public function clip(float $min,float $max): static
    { return $this->mapEl(fn($v)=>max($min,min($max,$v))); }
    public function clipLower(float $min): static { return $this->mapEl(fn($v)=>max($min,$v)); }
    public function clipUpper(float $max): static { return $this->mapEl(fn($v)=>min($max,$v)); }

    public function reduce(callable $fn, int|float $initial=0): int|float
    { $acc=$initial; foreach($this->a as $v) $acc=$fn($acc,$v); return $acc; }

    public function asType(int $dtype): static
    {
        $fn=$dtype===Tensor::INT?fn($v)=>(float)(int)$v:fn($v)=>(float)$v;
        return $this->mapEl($fn);
    }

    // ── arithmetic ────────────────────────────────────────────────────────

    public function add(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v+(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$v+$b->a[$i];
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function subtract(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v-(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$v-$b->a[$i];
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function multiply(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v*(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$v*$b->a[$i];
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function divide(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v/(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$v/$b->a[$i];
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function pow(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v**(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=pow($v,$b->a[$i]);
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function mod(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>fmod($v,(float)$b));
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=fmod($v,$b->a[$i]);
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    private function cmp(int|float|Tensor $b, callable $fn): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$fn($v,(float)$b)?1.0:0.0);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$fn($v,$b->a[$i])?1.0:0.0;
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function equal(int|float|Tensor $b): static       { return $this->cmp($b,fn($x,$y)=>$x==$y); }
    public function notEqual(int|float|Tensor $b): static    { return $this->cmp($b,fn($x,$y)=>$x!=$y); }
    public function greater(int|float|Tensor $b): static     { return $this->cmp($b,fn($x,$y)=>$x>$y); }
    public function greaterEqual(int|float|Tensor $b): static{ return $this->cmp($b,fn($x,$y)=>$x>=$y); }
    public function less(int|float|Tensor $b): static        { return $this->cmp($b,fn($x,$y)=>$x<$y); }
    public function lessEqual(int|float|Tensor $b): static   { return $this->cmp($b,fn($x,$y)=>$x<=$y); }

    // ── vector-specific ops ───────────────────────────────────────────────

    /** Dot product with another vector */
    public function dot(Vector $b): float
    {
        if ($b->n !== $this->n) throw new \RuntimeException("Length mismatch for dot product");
        $blas = Blas::getInstance();
        if ($blas->available()) return $blas->ddot($this->a, $b->a, $this->n);
        $s=0.0; foreach($this->a as $i=>$v) $s+=$v*$b->a[$i]; return $s;
    }

    public function inner(Vector $b): float { return $this->dot($b); }

    /** Outer product: this(n) × b(m) → Matrix(n×m) */
    public function outer(Vector $b): Matrix
    {
        $bArr = $b->asArray(); $m = $b->n(); $a=[];
        foreach($this->a as $i=>$vi) { $r=[]; foreach($bArr as $j=>$vj) $r[]=$vi*$vj; $a[$i]=$r; }
        return Matrix::quick($a);
    }

    /** 3D cross product */
    public function cross(Vector $b): static
    {
        if ($this->n!==3||$b->n!==3) throw new \RuntimeException("Cross product requires 3D vectors");
        $a=$this->a; $bA=$b->a;
        return new static([
            $a[1]*$bA[2]-$a[2]*$bA[1],
            $a[2]*$bA[0]-$a[0]*$bA[2],
            $a[0]*$bA[1]-$a[1]*$bA[0],
        ], 3);
    }

    /** p-norm */
    public function norm(float $p=2.0): float
    {
        if ($p===2.0) {
            $blas=Blas::getInstance();
            if ($blas->available()) return $blas->dnrm2($this->a,$this->n);
            $s=0.0; foreach($this->a as $v) $s+=$v*$v; return sqrt($s);
        }
        if ($p===1.0) { $s=0.0; foreach($this->a as $v) $s+=abs($v); return $s; }
        if (is_infinite($p)) { $m=0.0; foreach($this->a as $v) $m=max($m,abs($v)); return $m; }
        $s=0.0; foreach($this->a as $v) $s+=pow(abs($v),$p); return pow($s,1.0/$p);
    }

    /** Normalise to unit vector under p-norm */
    public function normalize(float $p=2.0): static
    {
        $n=$this->norm($p); if($n<1e-14) return clone $this;
        return $this->divide($n);
    }

    /** Vector projection of this onto b */
    public function project(Vector $b): static
    {
        $bn2 = $b->dot($b); if($bn2<1e-14) throw new \RuntimeException("Cannot project onto zero vector");
        return $b->multiply($this->dot($b)/$bn2);
    }

    /** Reflect this vector through the plane with normal b */
    public function reflect(Vector $b): static
    {
        return $this->subtract($b->multiply(2.0*$this->dot($b)/$b->dot($b)));
    }

    /** 1D convolution of this with kernel $b */
    public function convolve(Vector $b, int $stride=1): static
    {
        $kn=$b->n(); $on=(int)(($this->n-$kn)/$stride)+1;
        $bArr=$b->asArray(); $a=[];
        for($i=0;$i<$on;++$i){
            $s=0.0; for($k=0;$k<$kn;++$k) $s+=$this->a[$i*$stride+$k]*$bArr[$k];
            $a[]=$s;
        }
        return new static($a,$on);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  COLUMN VECTOR CLASS
// ═══════════════════════════════════════════════════════════════════════════

class ColumnVector extends Vector
{
    public function asVector(): Vector { return Vector::quick($this->a); }
    public function asMatrix(): Matrix
    { $a=[]; foreach($this->a as $v) $a[]= [$v]; return Matrix::quick($a); }

    public function asRowMatrix(): Matrix
    { return Matrix::quick([$this->a]); }

    public function asColumnMatrix(): Matrix
    { $a=[]; foreach($this->a as $v) $a[]= [$v]; return Matrix::quick($a); }
}

//  TENSOR DEMO & SELF-TEST
// ═══════════════════════════════════════════════════════════════════════════

function tensorTest(): void
{
    $blas = Blas::getInstance();
    $ffiOk = $blas->available();
    $lapackOk = $blas->hasLapack();

    printf("═══════════════════════════════════════════════════\n");
    printf("  Tensor Library Self-Test\n");
    printf("═══════════════════════════════════════════════════\n");
    printf("  FFI/BLAS   : %s\n", $ffiOk   ? '✓ available' : '✗ not found — pure PHP fallbacks active');
    printf("  LAPACK     : %s\n", $lapackOk ? '✓ available' : ($ffiOk ? '✗ symbols missing in loaded lib' : '✗ (FFI not loaded)'));
    if (!$ffiOk) {
        printf("  BLAS hint  : install libopenblas-dev, enable ext-ffi in php.ini\n");
        printf("              (ffi.enable=true or ffi.enable=preload)\n");
    }
    printf("───────────────────────────────────────────────────\n");

    $pass = 0; $fail = 0;
    $check = function(string $name, bool $ok, string $detail='') use (&$pass, &$fail) {
        if ($ok) { printf("  ✓ %-40s\n", $name); ++$pass; }
        else      { printf("  ✗ %-40s  %s\n", $name, $detail); ++$fail; }
    };

    // ── Matrix factories ──────────────────────────────────────────────

    $A = Matrix::build([[1,2],[3,4]]);
    $check('Matrix::build + shape', $A->shape() === [2,2]);
    $check('Matrix::zeros',  Matrix::zeros(3,2)->sum() === 0.0);
    $check('Matrix::ones',   Matrix::ones(2,3)->sum() === 6.0);
    $check('Matrix::fill',   Matrix::fill(5.0,2,2)->sum() === 20.0);
    $I = Matrix::eye(3);
    $check('Matrix::eye',    $I->trace() === 3.0 && $I->det() === 1.0);
    $D = Matrix::diagonal([1,2,3]);
    $check('Matrix::diagonal', $D->trace() === 6.0);

    // ── Element ops ───────────────────────────────────────────────────

    $B = Matrix::build([[1,4],[9,16]]);
    $check('Matrix::sqrt',   $B->sqrt()->asArray() === [[1.0,2.0],[3.0,4.0]]);
    $check('Matrix::square', Matrix::build([[1,2],[3,4]])->square()->asArray() === [[1.0,4.0],[9.0,16.0]]);
    $check('Matrix::negate', Matrix::build([[1,-2]])->negate()->asArray() === [[-1.0,2.0]]);
    $check('Matrix::abs',    Matrix::build([[-1,2],[-3,4]])->abs()->sum() === 10.0);
    $check('Matrix::clip',   Matrix::build([[0,5],[10,15]])->clip(2,8)->asArray() === [[2.0,5.0],[8.0,8.0]]);

    // ── Arithmetic ────────────────────────────────────────────────────

    $A = Matrix::build([[1,2],[3,4]]);
    $check('Matrix add scalar',  $A->add(1)->sum() === 14.0);
    $check('Matrix sub scalar',  $A->subtract(1)->sum() === 6.0);
    $check('Matrix mul scalar',  $A->multiply(2)->sum() === 20.0);
    $check('Matrix div scalar',  $A->divide(2)->asArray() === [[0.5,1.0],[1.5,2.0]]);
    $A2 = Matrix::build([[1,0],[0,1]]);
    $check('Matrix add matrix',  $A->add($A2)->asArray() === [[2.0,2.0],[3.0,5.0]]);

    // ── Reductions ────────────────────────────────────────────────────

    $A = Matrix::build([[1,2,3],[4,5,6]]);
    $check('rowSums',   $A->rowSums()->asArray() === [6.0,15.0]);
    $check('colSums',   $A->columnSums()->asArray() === [5.0,7.0,9.0]);
    $check('rowMeans',  $A->rowMeans()->asArray() === [2.0,5.0]);
    $check('colMeans',  $A->columnMeans()->asArray() === [2.5,3.5,4.5]);
    $check('rowMaxima', $A->rowMaxima()->asArray() === [3.0,6.0]);
    $check('rowMinima', $A->rowMinima()->asArray() === [1.0,4.0]);

    // ── Matmul ────────────────────────────────────────────────────────

    $A = Matrix::build([[1,2],[3,4]]);
    $B = Matrix::build([[5,6],[7,8]]);
    $C = $A->matmul($B);
    $expected = [[19.0,22.0],[43.0,50.0]];
    $ok = true;
    foreach ($C->asArray() as $i=>$row)
        foreach ($row as $j=>$v)
            if (abs($v-$expected[$i][$j])>1e-9) $ok=false;
    $check('Matrix::matmul 2×2', $ok, sprintf("got %s", $C));

    // ── Matmul large (benchmark) ──────────────────────────────────────

    $n   = 512;
    $M1  = Matrix::rand($n, $n);
    $M2  = Matrix::rand($n, $n);
    $t0  = microtime(true);
    $M3  = $M1->matmul($M2);
    $tMM = microtime(true) - $t0;
    $check("matmul {$n}×{$n} (using ".($ffiOk?'BLAS':'PHP').')', $M3->m() === $n && $M3->n() === $n);
    printf("    time: %.3f sec\n", $tMM);

    // ── Transpose ────────────────────────────────────────────────────

    $T = Matrix::build([[1,2,3],[4,5,6]])->transpose();
    $check('Matrix::transpose', $T->shape()===[3,2] && $T->asArray()[0]===[1.0,4.0]);

    // ── Dot (matrix × vector) ─────────────────────────────────────────

    $A = Matrix::build([[1,2],[3,4]]);
    $x = Vector::build([1,1]);
    $y = $A->dot($x);
    $check('Matrix::dot(Vector)', $y instanceof ColumnVector && $y->asArray()===[3.0,7.0]);

    // ── LU ───────────────────────────────────────────────────────────

    $A = Matrix::build([[2,1,1],[4,3,3],[8,7,9]]);
    ['L'=>$L,'U'=>$U,'P'=>$P] = $A->lu();
    $PA  = $P->matmul($A);
    $LU  = $L->matmul($U);
    $ok  = true;
    foreach ($PA->asArray() as $i=>$row)
        foreach ($row as $j=>$v)
            if (abs($v-$LU->asArray()[$i][$j])>1e-8) $ok=false;
    $check('Matrix::lu  P·A = L·U', $ok);

    // ── Inverse ──────────────────────────────────────────────────────

    $A   = Matrix::build([[2,1],[5,3]]);
    $inv = $A->inverse();
    $I2  = $A->matmul($inv);
    $ok  = abs($I2->asArray()[0][0]-1.0)<1e-9 && abs($I2->asArray()[0][1])<1e-9;
    $check('Matrix::inverse  A·A⁻¹ = I', $ok);

    // ── Det ──────────────────────────────────────────────────────────

    $check('Matrix::det 2×2', abs(Matrix::build([[3,8],[4,6]])->det()-(-14.0))<1e-9);

    // ── Solve ─────────────────────────────────────────────────────────

    $A = Matrix::build([[2,1],[-1,3]]);
    $b = Matrix::build([[5],[0]]);
    $x = $A->solve($b);
    $ok = abs($x->asArray()[0][0]-3.0)<1e-8 && abs($x->asArray()[1][0]-(-1.0))<1e-8;
    $check('Matrix::solve  Ax=b', $ok, sprintf("x=%s", $x));

    // ── Cholesky ──────────────────────────────────────────────────────

    $A = Matrix::build([[4,2],[2,3]]);
    $L = $A->cholesky();
    $LLt = $L->matmul($L->transpose());
    $ok  = abs($LLt->asArray()[0][0]-4.0)<1e-9 && abs($LLt->asArray()[1][0]-2.0)<1e-9;
    $check('Matrix::cholesky  L·Lᵀ = A', $ok);

    // ── SVD ───────────────────────────────────────────────────────────

    $A = Matrix::build([[1,0,0],[0,2,0],[0,0,3]]);
    ['U'=>$U,'s'=>$s,'VT'=>$VT] = $A->svd();
    $sv = $s->asArray(); rsort($sv);
    $check('Matrix::svd singular values', abs($sv[0]-3.0)<1e-6 && abs($sv[1]-2.0)<1e-6 && abs($sv[2]-1.0)<1e-6);

    // ── Eig (symmetric) ───────────────────────────────────────────────

    $A = Matrix::build([[2,-1],[-1,2]]);
    ['values'=>$ev] = $A->eig(true);
    $vals = $ev->asArray(); sort($vals);
    $check('Matrix::eig symmetric eigenvalues', abs($vals[0]-1.0)<1e-8 && abs($vals[1]-3.0)<1e-8);

    // ── REF / RREF ────────────────────────────────────────────────────

    $A = Matrix::build([[1,2,3],[4,5,6],[7,8,9]]);
    $check('Matrix::rank (singular)', $A->rank()===2);
    $R = $A->rref();
    $check('Matrix::rref top-left = 1', abs($R->asArray()[0][0]-1.0)<1e-9);

    // ── Vector ops ────────────────────────────────────────────────────

    $v1 = Vector::build([3,4]);
    $check('Vector::norm L2',  abs($v1->norm()-5.0)<1e-10);
    $check('Vector::norm L1',  abs($v1->norm(1.0)-7.0)<1e-10);
    $v2 = Vector::build([1,0]);
    $check('Vector::dot',      abs($v1->dot($v2)-3.0)<1e-10);
    $check('Vector::normalize', abs($v1->normalize()->norm()-1.0)<1e-10);

    $v3 = Vector::build([1,0,0]);
    $v4 = Vector::build([0,1,0]);
    $cross = $v3->cross($v4);
    $check('Vector::cross', $cross->asArray()===[0.0,0.0,1.0]);

    $outer = $v3->outer($v4);
    $check('Vector::outer shape', $outer->shape()===[3,3]);

    $cv = ColumnVector::build([1,2,3]);
    $check('ColumnVector instance', $cv instanceof ColumnVector && $cv instanceof Vector);
    $check('ColumnVector::asMatrix', $cv->asMatrix()->shape()===[3,1]);

    // ── Softmax ───────────────────────────────────────────────────────

    $A = Matrix::build([[1.0,2.0,3.0]]);
    $sm = $A->softmax()->asArray()[0];
    $check('Matrix::softmax sums to 1', abs(array_sum($sm)-1.0)<1e-12);

    // ── Comparisons ───────────────────────────────────────────────────

    $v = Vector::build([1,2,3,4,5]);
    $check('Vector::greater scalar', $v->greater(3)->asArray()===[0.0,0.0,0.0,1.0,1.0]);
    $check('Vector::lessEqual scalar', $v->lessEqual(3)->asArray()===[1.0,1.0,1.0,0.0,0.0]);

    // ── Summary ───────────────────────────────────────────────────────

    printf("───────────────────────────────────────────────────\n");
    printf("  Passed: %d / %d\n", $pass, $pass+$fail);
    if ($fail > 0) printf("  FAILED: %d\n", $fail);
    printf("═══════════════════════════════════════════════════\n");
}

// Only run self-test when this file is executed directly:
//   php Tensor.php
// When required by another script, nothing runs automatically.
if (PHP_SAPI === 'cli' && isset($argv[0]) && realpath($argv[0]) === realpath(__FILE__)) {
    tensorTest();
}