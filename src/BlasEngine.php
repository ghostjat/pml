<?php
declare(strict_types=1);

namespace Pml;

final class BlasEngine
{
    private static ?self $instance = null;
    public readonly \FFI $ffi;

    // We only need the Float32 (single precision) BLAS ops for LLM inference
    private const HEADER = '
        typedef enum { CblasRowMajor=101, CblasColMajor=102 } CBLAS_LAYOUT;
        typedef enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113 } CBLAS_TRANSPOSE;

        /* Level 1: Vector Ops */
        float cblas_sdot(int n, const float *x, int incx, const float *y, int incy);
        void  cblas_saxpy(int n, float alpha, const float *x, int incx, float *y, int incy);
        void  cblas_scopy(int n, const float *x, int incx, float *y, int incy);
        void  cblas_sscal(int n, float alpha, float *x, int incx);

        /* Level 2: Matrix-Vector Ops */
        void cblas_sgemv(int Order, int TransA, int M, int N,
                         float alpha, const float *A, int lda,
                         const float *x, int incx,
                         float beta,  float *y, int incy);

        /* Level 3: Matrix-Matrix Ops (The Heavy Lifters for QKV & Attention) */
        void cblas_sgemm(int Order, int TransA, int TransB,
                         int M, int N, int K,
                         float alpha, const float *A, int lda,
                         const float *B, int ldb,
                         float beta,  float *C, int ldc);
    ';

    private function __construct()
    {
        if (!extension_loaded('ffi')) {
            throw new \RuntimeException("FFI extension is required.");
        }

        // Standard lookup for OpenBLAS (Ubuntu/Debian, macOS, etc.)
        $candidates = [
            '/usr/lib/x86_64-linux-gnu/libopenblas.so.0',
            '/usr/lib/aarch64-linux-gnu/libopenblas.so.0',
            '/opt/homebrew/opt/openblas/lib/libopenblas.dylib', // Apple Silicon
            'libopenblas.so.0',
            'libopenblas.so'
        ];

        foreach ($candidates as $lib) {
            try {
                $this->ffi = \FFI::cdef(self::HEADER, $lib);
                return;
            } catch (\FFI\Exception $e) {
                continue;
            }
        }
        throw new \RuntimeException("Could not load OpenBLAS shared library.");
    }

    public static function get(): self
    {
        return self::$instance ??= new self();
    }
}