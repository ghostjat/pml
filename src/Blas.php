<?php
declare(strict_types=1);

/**
 * Pharos — FFI-backed linear algebra for PHP
 * ============================================
 * Drop-in compatible with RubixML/Tensor (same namespace, same signatures).
 *
 * BACKEND SELECTION (automatic, per-operation):
 *   1. BLAS/LAPACK via FFI (libopenblas / libcblas / liblapacke)
 *      — used for: matmul, dot, norm, inverse, solve, svd, eig, cholesky, lu
 *   2. Pure PHP — used for all ops when FFI unavailable, and for all
 *      element-wise / reduction operations regardless
 *
 * STORAGE: PHP arrays (immutable, serialisable, parallel-compatible).
 *   FFI buffers are allocated transiently per-operation and freed immediately.
 *
 * COMPATIBILITY:
 *   namespace Tensor;  classes Matrix, Vector, ColumnVector, interface Tensor
 *   All factory methods static.  All ops return new instances.
 *   Implements Countable, IteratorAggregate on Matrix and Vector.
 */

namespace Pml;

if (defined('_RUBIX_TENSOR_FFI_LOADED_')) return;
define('_RUBIX_TENSOR_FFI_LOADED_', true);

// ═══════════════════════════════════════════════════════════════════════════
//  BLAS / LAPACK FFI BACKEND
// ═══════════════════════════════════════════════════════════════════════════

final class Blas
{
    private static ?self $instance = null;
    private ?\FFI $ffi = null;
    private bool  $hasLapack = true;

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
        foreach (self::CANDIDATES as $lib) {
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
