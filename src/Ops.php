<?php

declare(strict_types=1);
namespace Pml;


class Ops
{
    /**
     * C = alpha * A * B
     * Operates exclusively on FFI pointers. Zero data is marshaled back to PHP.
     */
    public static function matmul(Tensor $A, Tensor $B, bool $transA = false, bool $transB = false): Tensor
    {
        // Calculate dimensions based on transpose flags
        $M = $transA ? $A->shape[1] : $A->shape[0];
        $K = $transA ? $A->shape[0] : $A->shape[1];
        $N = $transB ? $B->shape[0] : $B->shape[1];

        if (($transB ? $B->shape[1] : $B->shape[0]) !== $K) {
            throw new \InvalidArgumentException("Inner matrix dimensions must agree.");
        }

        $C = new Tensor([$M, $N]);
        $ffi = BlasEngine::get()->ffi;

        $cblasTransA = $transA ? 112 : 111; // CblasTrans : CblasNoTrans
        $cblasTransB = $transB ? 112 : 111;

        $lda = $A->shape[1];
        $ldb = $B->shape[1];
        $ldc = $N; // C is always row-major and untransposed in memory

        $ffi->cblas_sgemm(
            101, // CblasRowMajor
            $cblasTransA, $cblasTransB,
            $M, $N, $K,
            1.0, 
            $A->buffer, $lda,
            $B->buffer, $ldb,
            0.0, 
            $C->buffer, $ldc
        );

        return $C;
    }

    /**
     * Adds vector bias to matrix Y in place: Y = Y + bias
     */
   public static function addInPlace(Tensor $target, Tensor $source): void
    {
        $ffi = BlasEngine::get()->ffi;
        $ffi->cblas_saxpy($target->size, 1.0, $source->buffer, 1, $target->buffer, 1);
    }
    
    /** Applied along the last dimension (rows) */
    public static function softmax(Tensor $X): void
    {
        $rows = $X->shape[0];
        $cols = $X->shape[1];
        
        for ($i = 0; $i < $rows; $i++) {
            $offset = $i * $cols;
            
            // Find max for numerical stability
            $max = -INF;
            for ($j = 0; $j < $cols; $j++) {
                if ($X->buffer[$offset + $j] > $max) $max = $X->buffer[$offset + $j];
            }
            
            // Exponentiate and sum
            $sum = 0.0;
            for ($j = 0; $j < $cols; $j++) {
                $val = exp($X->buffer[$offset + $j] - $max);
                $X->buffer[$offset + $j] = $val;
                $sum += $val;
            }
            
            // Normalize
            for ($j = 0; $j < $cols; $j++) {
                $X->buffer[$offset + $j] /= $sum;
            }
        }
    }

    /** Root Mean Square Normalization (used in LLaMA/Mistral) */
    public static function rmsNorm(Tensor $X, Tensor $weight, float $eps = 1e-5): void
    {
        $rows = $X->shape[0];
        $cols = $X->shape[1];
        
        for ($i = 0; $i < $rows; $i++) {
            $offset = $i * $cols;
            $ss = 0.0;
            
            for ($j = 0; $j < $cols; $j++) {
                $val = $X->buffer[$offset + $j];
                $ss += $val * $val;
            }
            
            $invNorm = 1.0 / sqrt(($ss / $cols) + $eps);
            
            for ($j = 0; $j < $cols; $j++) {
                $X->buffer[$offset + $j] *= $invNorm * $weight->buffer[$j];
            }
        }
    }
}