<?php

declare(strict_types=1);

namespace Pml;

/**
 * BlasEngine: A singleton FFI bridge to OpenBLAS + LAPACKE.
 *
 * Design: Single engine instance shared across the entire process.
 * CBLAS (cblas_s*) is loaded from OpenBLAS; LAPACKE (LAPACKE_s*) is loaded
 * from a separate liblapacke when OpenBLAS does not bundle it.
 */
final class BlasEngine
{
    private static ?self $instance = null;
    public readonly \FFI $ffi;
    public readonly \FFI $lapacke;

    // ── FFI Header: Float32 CBLAS (Level 1/2/3) ───────────────────────────
    private const BLAS_HEADER = <<<'C'
        /* ── CBLAS Enumerations ─────────────────────────────────────────── */
        typedef enum { CblasRowMajor=101, CblasColMajor=102 }        CBLAS_LAYOUT;
        typedef enum { CblasNoTrans=111, CblasTrans=112 }             CBLAS_TRANSPOSE;
        typedef enum { CblasUpper=121, CblasLower=122 }               CBLAS_UPLO;
        typedef enum { CblasNonUnit=131, CblasUnit=132 }              CBLAS_DIAG;
        typedef enum { CblasLeft=141, CblasRight=142 }                CBLAS_SIDE;

        /* ── Level 1: Vector ────────────────────────────────────────────── */
        float cblas_sdot  (int n, const float *x, int incx, const float *y, int incy);
        float cblas_snrm2 (int n, const float *x, int incx);
        float cblas_sasum (int n, const float *x, int incx);
        int   cblas_isamax(int n, const float *x, int incx);
        void  cblas_saxpy (int n, float alpha, const float *x, int incx, float *y, int incy);
        void  cblas_scopy (int n, const float *x, int incx, float *y, int incy);
        void  cblas_sscal (int n, float alpha, float *x, int incx);
        void  cblas_sswap (int n, float *x, int incx, float *y, int incy);

        /* ── Level 2: Matrix-Vector ─────────────────────────────────────── */
        void cblas_sgemv(int Order, int TransA, int M, int N,
                         float alpha, const float *A, int lda,
                         const float *x, int incx,
                         float beta,  float *y, int incy);
        void cblas_ssymv(int Order, int Uplo, int N,
                         float alpha, const float *A, int lda,
                         const float *x, int incx,
                         float beta,  float *y, int incy);
        void cblas_strmv(int Order, int Uplo, int TransA, int Diag,
                         int N, const float *A, int lda, float *x, int incx);
        void cblas_sger (int Order, int M, int N,
                         float alpha, const float *x, int incx,
                         const float *y, int incy, float *A, int lda);

        /* ── Level 3: Matrix-Matrix ─────────────────────────────────────── */
        void cblas_sgemm(int Order, int TransA, int TransB,
                         int M, int N, int K,
                         float alpha, const float *A, int lda,
                         const float *B, int ldb,
                         float beta,  float *C, int ldc);
        void cblas_ssymm(int Order, int Side, int Uplo, int M, int N,
                         float alpha, const float *A, int lda,
                         const float *B, int ldb,
                         float beta,  float *C, int ldc);
        void cblas_strmm(int Order, int Side, int Uplo, int TransA, int Diag,
                         int M, int N,
                         float alpha, const float *A, int lda, float *B, int ldb);

        /* ── Memory Allocation Helpers ──────────────────────────────────── */
        void *malloc(size_t size);
        void  free(void *ptr);
    C;

    // ── FFI Header: LAPACKE ───────────────────────────────────────────────
    private const LAPACKE_HEADER = <<<'C'
        int LAPACKE_sgetrf(int matrix_layout, int m, int n, float *a, int lda, int *ipiv);
        int LAPACKE_sgetri(int matrix_layout, int n, float *a, int lda, const int *ipiv);
        int LAPACKE_sgetrs(int matrix_layout, char trans, int n, int nrhs,
                           const float *a, int lda, const int *ipiv, float *b, int ldb);
        int LAPACKE_sposv (int matrix_layout, char uplo, int n, int nrhs,
                           float *a, int lda, float *b, int ldb);
        int LAPACKE_ssyev (int matrix_layout, char jobz, char uplo, int n,
                           float *a, int lda, float *w);
        int LAPACKE_sgesvd(int matrix_layout, char jobu, char jobvt, int m, int n,
                           float *a, int lda, float *s, float *u, int ldu,
                           float *vt, int ldvt, float *superb);
        int LAPACKE_sgels (int matrix_layout, char trans, int m, int n, int nrhs,
                           float *a, int lda, float *b, int ldb);
    C;

    // ── OpenBLAS shared library search paths ──────────────────────────────
    private const BLAS_CANDIDATES = [
        '/usr/lib/x86_64-linux-gnu/libopenblas.so.0',
        '/usr/lib/x86_64-linux-gnu/libopenblas64.so.0',
        '/usr/lib/aarch64-linux-gnu/libopenblas.so.0',
        '/usr/lib/aarch64-linux-gnu/libopenblas64.so.0',
        '/opt/homebrew/opt/openblas/lib/libopenblas.dylib',   // macOS Apple Silicon
        '/usr/local/opt/openblas/lib/libopenblas.dylib',       // macOS Intel Homebrew
        '/usr/lib/libopenblas.so.0',
        'libopenblas.so.0',
        'libopenblas.so',
    ];

    // ── LAPACKE shared library search paths ───────────────────────────────
    private const LAPACKE_CANDIDATES = [
        '/usr/lib/x86_64-linux-gnu/liblapacke.so.3',
        '/usr/lib/x86_64-linux-gnu/liblapacke.so',
        '/usr/lib/aarch64-linux-gnu/liblapacke.so.3',
        '/usr/lib/aarch64-linux-gnu/liblapacke.so',
        '/usr/lib/liblapacke.so.3',
        '/usr/lib/liblapacke.so',
        'liblapacke.so.3',
        'liblapacke.so',
    ];

    private function __construct()
    {
        if (!extension_loaded('ffi')) {
            throw new \RuntimeException(
                'Pml requires the FFI extension. Enable it in your php.ini: extension=ffi'
            );
        }

        // Load CBLAS from OpenBLAS — also try it for LAPACKE first (bundled builds)
        $blasFfi = null;
        foreach (self::BLAS_CANDIDATES as $lib) {
            try {
                $blasFfi = \FFI::cdef(self::BLAS_HEADER, $lib);
                break;
            } catch (\FFI\Exception) {
                continue;
            }
        }

        if ($blasFfi === null) {
            throw new \RuntimeException(
                "Pml could not load OpenBLAS.\n"
                . "Install it with: apt-get install libopenblas-dev  (Linux)\n"
                . "                  brew install openblas             (macOS)\n"
                . "Searched: " . implode(', ', self::BLAS_CANDIDATES)
            );
        }

        $this->ffi = $blasFfi;

        // Try LAPACKE from OpenBLAS first (bundled), then fall back to liblapacke
        $lapackeFfi = null;
        foreach (array_merge(self::BLAS_CANDIDATES, self::LAPACKE_CANDIDATES) as $lib) {
            try {
                $lapackeFfi = \FFI::cdef(self::LAPACKE_HEADER, $lib);
                break;
            } catch (\FFI\Exception) {
                continue;
            }
        }

        if ($lapackeFfi === null) {
            throw new \RuntimeException(
                "Pml could not load LAPACKE.\n"
                . "Install it with: apt-get install liblapacke-dev  (Linux)\n"
                . "                  brew install lapack              (macOS)\n"
                . "Searched: " . implode(', ', array_merge(self::BLAS_CANDIDATES, self::LAPACKE_CANDIDATES))
            );
        }

        $this->lapacke = $lapackeFfi;
    }

    public static function get(): self
    {
        return self::$instance ??= new self();
    }

    /**
     * Convenience: allocate a zeroed float[N] CData buffer.
     * GC-owned (owned=true) — PHP will free on destruction.
     */
    public function allocFloat(int $n, bool $owned = true): \FFI\CData
    {
        return $this->ffi->new("float[{$n}]", $owned);
    }

    /**
     * Convenience: allocate a zeroed int[N] CData buffer (for LAPACK pivots).
     */
    public function allocInt(int $n, bool $owned = true): \FFI\CData
    {
        return $this->ffi->new("int[{$n}]", $owned);
    }

    /** LAPACKE row-major layout constant */
    public const LAPACK_ROW_MAJOR = 101;
    public const LAPACK_COL_MAJOR = 102;
}