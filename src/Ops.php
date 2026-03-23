<?php

declare(strict_types=1);

namespace Pml;

/**
 * Ops: The Math Engine.
 *
 * All operations maintain the zero-copy / FFI-first philosophy:
 * - BLAS operations consume FFI buffers directly via cblas_s* calls.
 * - LAPACK operations consume FFI buffers directly via LAPACKE_s* calls.
 * - Element-wise loops in PHP are used only where BLAS/LAPACK have no equivalent.
 * - No operation marshals data to PHP arrays internally.
 */
final class Ops
{
    // ── Matrix Multiplication (BLAS sgemm) ────────────────────────────────

    /**
     * Matrix multiply: C = alpha * op(A) * op(B) + beta * C
     *
     * Handles the logical transpose flag on Tensor::T() — no data is ever moved.
     * Supports 2D tensors only. For batched matmul, use batchMatmul().
     *
     * @param bool $transA  Force transpose A (overrides A->_transposed)
     * @param bool $transB  Force transpose B (overrides B->_transposed)
     */
    public static function matmul(
        Tensor $A,
        Tensor $B,
        bool   $transA = false,
        bool   $transB = false,
        float  $alpha = 1.0,
        float  $beta  = 0.0,
        ?Tensor $C    = null
    ): Tensor {
        // ── JIT Int8 Dequantisation ────────────────────────────────────────
        // Keep references to the ORIGINAL tensors for the autograd graph.
        // INT8 tensors have requiresGrad=false, so the backward will skip
        // them automatically — no gradient flows through quantised weights.
        $origA = $A;
        $origB = $B;
        if ($A->dtype === Tensor::INT8) { $A = $A->dequantize(); }
        if ($B->dtype === Tensor::INT8) { $B = $B->dequantize(); }

        // ── Resolve effective transpose flags ─────────────────────────────
        // Tensor::T() sets _transposed=true; the $transA/$transB parameters
        // provide an alternative: pass the weight matrix directly and set the
        // flag here.  Both paths set the BLAS CblasTrans/CblasNoTrans bit.
        $effTransA = $transA || $A->_transposed;
        $effTransB = $transB || $B->_transposed;

        // Effective (mathematical) dimensions of op(A)[M,K] and op(B)[K,N].
        // When _transposed is set, the stored shape is already reversed by T(),
        // so we reverse it back to get the physical [rows, cols].
        [$Arows, $Acols] = $A->_transposed
            ? [$A->shape[1], $A->shape[0]]
            : [$A->shape[0], $A->shape[1]];
        [$Brows, $Bcols] = $B->_transposed
            ? [$B->shape[1], $B->shape[0]]
            : [$B->shape[0], $B->shape[1]];

        $M  = $effTransA ? $Acols : $Arows;
        $K  = $effTransA ? $Arows : $Acols;
        $N  = $effTransB ? $Brows : $Bcols;
        $K2 = $effTransB ? $Bcols : $Brows;

        if ($K !== $K2) {
            throw new \InvalidArgumentException(
                "matmul: inner dimensions mismatch. op(A) has K={$K}, op(B) has K={$K2}."
            );
        }

        $out = $C ?? new Tensor([$M, $N]);

        // Leading dimensions are always the PHYSICAL column count.
        $lda = $A->shape[1];
        $ldb = $B->shape[1];
        $ldc = $N;

        BlasEngine::get()->ffi->cblas_sgemm(
            101,                              // CblasRowMajor
            $effTransA ? 112 : 111,           // CblasTrans / CblasNoTrans for A
            $effTransB ? 112 : 111,           // CblasTrans / CblasNoTrans for B
            $M, $N, $K,
            $alpha,
            $A->buffer, $lda,
            $B->buffer, $ldb,
            $beta,
            $out->buffer, $ldc
        );

        // ── Build computational graph (autograd) ─────────────────────────
        // Only pay the graph-construction cost when at least one input is a
        // learnable parameter (requiresGrad=true).  During pure inference
        // everything is false and this block is a no-op.
        if ($origA->requiresGrad || $origB->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$origA, $origB];

            // Physical column counts — BLAS "leading dimension" for each buffer.
            //
            // For op(A)[M,K]:
            //   !effTransA → A_phys is [M,K], physical cols = K
            //    effTransA → A_phys is [K,M], physical cols = M
            // For op(B)[K,N]:
            //   !effTransB → B_phys is [K,N], physical cols = N
            //    effTransB → B_phys is [N,K], physical cols = K
            $physACols = $effTransA ? $M : $K;
            $physBCols = $effTransB ? $K : $N;

            // Snapshot values the closure needs; closures are long-lived.
            $fA = $effTransA;
            $fB = $effTransB;
            $_M = $M;
            $_K = $K;
            $_N = $N;

            // ── _backward closure ─────────────────────────────────────────
            //
            // Forward:  C[M,N]   = op(A)[M,K] @ op(B)[K,N]
            //
            // Backward rules (chain rule through matrix multiply):
            //
            //   dL/d(op(A))[M,K] = dC[M,N] @ op(B)^T[N,K]
            //   dL/d(op(B))[K,N] = op(A)^T[K,M] @ dC[M,N]
            //
            // When an operand was PHYSICALLY transposed (effTransA/B), the
            // stored buffer has shape [K,M] or [N,K] respectively.  We must
            // further transpose the effective gradient back to match the
            // physical storage layout before accumulating.
            //
            // All four combinations are covered below:
            //
            //  Case ─────── dA_phys computation (BLAS call) ──────────────────────────────────────
            //  !fA !fB   dA[M,K] = dC[M,N] @ B_phys[K,N]^T      → sgemm(NT,T,  M,K,N, dC,N, B,N)
            //  !fA  fB   dA[M,K] = dC[M,N] @ B_phys[N,K]        → sgemm(NT,NT, M,K,N, dC,N, B,K)
            //   fA !fB   dA[K,M] = B_phys[K,N] @ dC[M,N]^T      → sgemm(NT,T,  K,M,N, B,N,  dC,N)
            //   fA  fB   dA[K,M] = B_phys[N,K]^T @ dC[M,N]^T    → sgemm(T, T,  K,M,N, B,K,  dC,N)
            //
            //  Case ─────── dB_phys computation (BLAS call) ──────────────────────────────────────
            //  !fA !fB   dB[K,N] = A_phys[M,K]^T @ dC[M,N]      → sgemm(T, NT, K,N,M, A,K,  dC,N)
            //   fA !fB   dB[K,N] = A_phys[K,M]   @ dC[M,N]      → sgemm(NT,NT, K,N,M, A,M,  dC,N)
            //  !fA  fB   dB[N,K] = dC[M,N]^T @ A_phys[M,K]      → sgemm(T, NT, N,K,M, dC,N, A,K)
            //   fA  fB   dB[N,K] = dC[M,N]^T @ A_phys[K,M]^T    → sgemm(T, T,  N,K,M, dC,N, A,M)
            //
            // All calls accumulate (beta=1.0) so gradients from multiple uses
            // of the same parameter are correctly summed.

            $out->_backward = static function()
                use ($origA, $origB, $A, $B, $out,
                     $fA, $fB, $_M, $_K, $_N, $physACols, $physBCols): void
            {
                $ffi = BlasEngine::get()->ffi;
                $dC  = $out->grad; // incoming gradient: shape [M,N], ldc=N

                // ── Gradient for A ────────────────────────────────────────
                if ($origA->requiresGrad) {
                    $origA->initGrad();

                    if (!$fA) {
                        // dA_phys[M,K] = dC[M,N] @ op(B)^T[N,K]
                        // op(B)^T = B_phys^T if !fB, else B_phys
                        $ffi->cblas_sgemm(
                            101,           // RowMajor
                            111,           // NoTrans  — dC
                            $fB ? 111 : 112, // NoTrans B_phys (already gives K cols) OR Trans
                            $_M, $_K, $_N,
                            1.0,
                            $dC, $_N,                         // lda = N (dC cols)
                            $B->buffer, $physBCols,           // ldb = physBCols
                            1.0,                              // accumulate
                            $origA->grad, $physACols          // ldc = K (!fA)
                        );
                    } else {
                        // dA_phys[K,M] = op(B)[K,N] @ dC^T[N,M]
                        // op(B) = B_phys if !fB, else B_phys^T
                        $ffi->cblas_sgemm(
                            101,
                            $fB ? 112 : 111, // Trans B_phys → op(B) when fB, else NoTrans
                            112,             // Trans  — dC
                            $_K, $_M, $_N,
                            1.0,
                            $B->buffer, $physBCols,
                            $dC, $_N,
                            1.0,
                            $origA->grad, $physACols          // ldc = M (fA)
                        );
                    }
                }

                // ── Gradient for B ────────────────────────────────────────
                if ($origB->requiresGrad) {
                    $origB->initGrad();

                    if (!$fB) {
                        // dB_phys[K,N] = op(A)^T[K,M] @ dC[M,N]
                        // op(A)^T = A_phys^T if !fA, else A_phys
                        $ffi->cblas_sgemm(
                            101,
                            $fA ? 111 : 112, // NoTrans A_phys (already K×M) OR Trans
                            111,             // NoTrans — dC
                            $_K, $_N, $_M,
                            1.0,
                            $A->buffer, $physACols,
                            $dC, $_N,
                            1.0,
                            $origB->grad, $physBCols          // ldc = N (!fB)
                        );
                    } else {
                        // dB_phys[N,K] = dC^T[N,M] @ op(A)[M,K]
                        // op(A) = A_phys^T if !fA, else A_phys
                        $ffi->cblas_sgemm(
                            101,
                            112,             // Trans — dC
                            $fA ? 112 : 111, // Trans A_phys → op(A) when fA, else NoTrans
                            $_N, $_K, $_M,
                            1.0,
                            $dC, $_N,
                            $A->buffer, $physACols,
                            1.0,
                            $origB->grad, $physBCols          // ldc = K (fB)
                        );
                    }
                }
            };
        }

        return $out;
    }

    /**
     * Batched matrix multiply: out[b] = A[b] * B[b]
     * A: [batch, M, K], B: [batch, K, N] → out: [batch, M, N]
     *
     * Implemented as a loop of sgemm calls on row-slices.
     * For very large batches, a single cblas_sgemm with a reshaped view is preferable.
     */
    public static function batchMatmul(Tensor $A, Tensor $B): Tensor
    {
        if (count($A->shape) !== 3 || count($B->shape) !== 3) {
            throw new \InvalidArgumentException('batchMatmul requires rank-3 tensors.');
        }
        [$batch, $M, $K] = $A->shape;
        [, $K2, $N]      = $B->shape;
        if ($K !== $K2)  throw new \InvalidArgumentException('batchMatmul: K mismatch.');
        if ($batch !== $B->shape[0]) throw new \InvalidArgumentException('batchMatmul: batch mismatch.');

        $out    = new Tensor([$batch, $M, $N]);
        $aSlice = $M * $K;
        $bSlice = $K * $N;
        $cSlice = $M * $N;
        $ffi    = BlasEngine::get()->ffi;

        for ($b = 0; $b < $batch; $b++) {
            $aPtr = \FFI::cast('float*', \FFI::addr($A->buffer[$b * $aSlice]));
            $bPtr = \FFI::cast('float*', \FFI::addr($B->buffer[$b * $bSlice]));
            $cPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$b * $cSlice]));

            $ffi->cblas_sgemm(101, 111, 111, $M, $N, $K, 1.0, $aPtr, $K, $bPtr, $N, 0.0, $cPtr, $N);
        }

        return $out;
    }

    /**
     * Matrix-vector multiply: y = alpha * op(A) * x + beta * y
     * A: [M, N], x: [N] → y: [M]
     */
    public static function matvec(Tensor $A, Tensor $x, bool $transA = false, float $alpha = 1.0, float $beta = 0.0): Tensor
    {
        $M = $A->shape[0];
        $N = $A->shape[1];
        $y = new Tensor([$transA ? $N : $M]);

        BlasEngine::get()->ffi->cblas_sgemv(
            101,
            $transA ? 112 : 111,
            $M, $N,
            $alpha, $A->buffer, $N,
            $x->buffer, 1,
            $beta, $y->buffer, 1
        );
        return $y;
    }

    // ── Element-wise BLAS Ops ─────────────────────────────────────────────

    /**
     * y += alpha * x  (BLAS saxpy — the standard "add with scale" primitive)
     * Both tensors must have the same size.
     */
    public static function saxpy(Tensor $x, Tensor $y, float $alpha = 1.0): void
    {
        if ($x->size !== $y->size) {
            throw new \InvalidArgumentException('saxpy: tensor sizes must match.');
        }
        BlasEngine::get()->ffi->cblas_saxpy($x->size, $alpha, $x->buffer, 1, $y->buffer, 1);
    }

    /**
     * Add: C = A + B (element-wise). Returns new tensor.
     *
     * Forward:   C[i] = A[i] + B[i]
     *
     * Backward (chain rule, addition distributes gradient equally):
     *   dL/dA[i] += dL/dC[i]    (saxpy alpha=1.0)
     *   dL/dB[i] += dL/dC[i]    (saxpy alpha=1.0)
     *
     * A and B must have identical sizes (no broadcasting).
     */
    public static function add(Tensor $A, Tensor $B): Tensor
    {
        $C = $A->clone();
        self::saxpy($B, $C, 1.0);

        if ($A->requiresGrad || $B->requiresGrad) {
            $C->requiresGrad = true;
            $C->_prev        = [$A, $B];

            $C->_backward = static function() use ($A, $B, $C): void {
                $ffi = BlasEngine::get()->ffi;

                // dA += dC  — gradient flows through addition unchanged
                if ($A->requiresGrad) {
                    $A->initGrad();
                    $ffi->cblas_saxpy($C->size, 1.0, $C->grad, 1, $A->grad, 1);
                }

                // dB += dC  — same gradient, same alpha
                if ($B->requiresGrad) {
                    $B->initGrad();
                    $ffi->cblas_saxpy($C->size, 1.0, $C->grad, 1, $B->grad, 1);
                }
            };
        }

        return $C;
    }

    /**
     * Subtract: C = A - B.
     */
    public static function sub(Tensor $A, Tensor $B): Tensor
    {
        $C = $A->clone();
        self::saxpy($B, $C, -1.0);
        return $C;
    }

    /**
     * Dot product of two 1D tensors.
     */
    public static function dot(Tensor $A, Tensor $B): float
    {
        if ($A->size !== $B->size) {
            throw new \InvalidArgumentException('dot: size mismatch.');
        }
        return (float) BlasEngine::get()->ffi->cblas_sdot($A->size, $A->buffer, 1, $B->buffer, 1);
    }

    /**
     * Hadamard (element-wise) multiply: C = A ⊙ B.
     */
    public static function mul(Tensor $A, Tensor $B): Tensor
    {
        if ($A->size !== $B->size) {
            throw new \InvalidArgumentException('mul: size mismatch.');
        }
        $C = new Tensor($A->shape);
        for ($i = 0; $i < $A->size; $i++) {
            $C->buffer[$i] = $A->buffer[$i] * $B->buffer[$i];
        }
        return $C;
    }

    /**
     * Element-wise divide: C = A / B.
     */
    public static function div(Tensor $A, Tensor $B): Tensor
    {
        if ($A->size !== $B->size) {
            throw new \InvalidArgumentException('div: size mismatch.');
        }
        $C = new Tensor($A->shape);
        for ($i = 0; $i < $A->size; $i++) {
            $C->buffer[$i] = $A->buffer[$i] / $B->buffer[$i];
        }
        return $C;
    }

    /**
     * Outer product: A[M] ⊗ B[N] → C[M, N].
     * Uses BLAS sger: A = alpha * x * y^T + A
     */
    public static function outer(Tensor $x, Tensor $y): Tensor
    {
        $M = $x->size;
        $N = $y->size;
        $C = Tensor::zeros([$M, $N]);
        BlasEngine::get()->ffi->cblas_sger(101, $M, $N, 1.0, $x->buffer, 1, $y->buffer, 1, $C->buffer, $N);
        return $C;
    }

    // ── Activation Functions ──────────────────────────────────────────────

    /**
     * ReLU in-place: x = max(0, x)
     */
    public static function reluInPlace(Tensor $x): void
    {
        for ($i = 0; $i < $x->size; $i++) {
            if ($x->buffer[$i] < 0.0) $x->buffer[$i] = 0.0;
        }
    }

    /**
     * ReLU — returns new tensor.
     */
    public static function relu(Tensor $x): Tensor
    {
        $out = $x->clone();
        self::reluInPlace($out);
        return $out;
    }

    /**
     * GELU (Gaussian Error Linear Unit) — approximation used in GPT/BERT.
     * GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
     */
    public static function gelu(Tensor $x): Tensor
    {
        $out = new Tensor($x->shape);
        $c   = sqrt(2.0 / M_PI);
        for ($i = 0; $i < $x->size; $i++) {
            $xi          = (float)$x->buffer[$i];
            $inner       = $c * ($xi + 0.044715 * $xi * $xi * $xi);
            $out->buffer[$i] = 0.5 * $xi * (1.0 + tanh($inner));
        }
        return $out;
    }

    /**
     * SiLU / Swish: x * sigmoid(x) — used in LLaMA/Mistral FFN.
     */
    public static function silu(Tensor $x): Tensor
    {
        $out = new Tensor($x->shape);
        for ($i = 0; $i < $x->size; $i++) {
            $xi              = (float)$x->buffer[$i];
            $out->buffer[$i] = $xi / (1.0 + exp(-$xi));
        }
        return $out;
    }

    /**
     * Sigmoid: 1 / (1 + e^-x)
     */
    public static function sigmoid(Tensor $x): Tensor
    {
        $out = new Tensor($x->shape);
        for ($i = 0; $i < $x->size; $i++) {
            $out->buffer[$i] = 1.0 / (1.0 + exp(-(float)$x->buffer[$i]));
        }
        return $out;
    }

    /**
     * Tanh element-wise.
     */
    public static function tanh(Tensor $x): Tensor
    {
        $out = new Tensor($x->shape);
        for ($i = 0; $i < $x->size; $i++) {
            $out->buffer[$i] = tanh((float)$x->buffer[$i]);
        }
        return $out;
    }

    // ── Normalization ─────────────────────────────────────────────────────

    /**
     * Softmax along last dimension (rows of a 2D tensor, or all of 1D).
     * Numerically stable: subtracts max before exp.
     * In-place.
     */
    public static function softmaxInPlace(Tensor $x): void
    {
        if (count($x->shape) === 1) {
            self::softmax1DInPlace($x->buffer, 0, $x->size);
            return;
        }

        $rows = (int)($x->size / $x->shape[count($x->shape) - 1]);
        $cols = $x->shape[count($x->shape) - 1];

        for ($i = 0; $i < $rows; $i++) {
            self::softmax1DInPlace($x->buffer, $i * $cols, $cols);
        }
    }

    /**
     * Softmax — returns new tensor.
     */
    public static function softmax(Tensor $x): Tensor
    {
        $out = $x->clone();
        self::softmaxInPlace($out);
        return $out;
    }

    /**
     * Log-softmax (numerically stable). Useful for cross-entropy loss.
     */
    public static function logSoftmax(Tensor $x): Tensor
    {
        $out  = $x->clone();
        $cols = $x->shape[count($x->shape) - 1];
        $rows = $x->size / $cols;

        for ($i = 0; $i < $rows; $i++) {
            $offset = $i * $cols;
            $maxVal = -INF;
            for ($j = 0; $j < $cols; $j++) {
                if ($out->buffer[$offset + $j] > $maxVal) $maxVal = $out->buffer[$offset + $j];
            }
            $logSum = 0.0;
            for ($j = 0; $j < $cols; $j++) {
                $logSum += exp($out->buffer[$offset + $j] - $maxVal);
            }
            $logSum = log($logSum) + $maxVal;
            for ($j = 0; $j < $cols; $j++) {
                $out->buffer[$offset + $j] -= $logSum;
            }
        }
        return $out;
    }

    /**
     * RMSNorm: used in LLaMA, Mistral, Gemma.
     * norm(x) = x / sqrt(mean(x²) + eps) * weight
     *
     * Operates row-by-row on a 2D tensor.
     * In-place: modifies $x.
     */
    public static function rmsNormInPlace(Tensor $x, Tensor $weight, float $eps = 1e-5): void
    {
        if (count($x->shape) < 2) {
            throw new \InvalidArgumentException('rmsNorm requires at least 2D tensor.');
        }
        $cols = $x->shape[count($x->shape) - 1];
        $rows = $x->size / $cols;

        for ($i = 0; $i < $rows; $i++) {
            $offset = $i * $cols;
            $ss     = 0.0;
            for ($j = 0; $j < $cols; $j++) {
                $v  = (float)$x->buffer[$offset + $j];
                $ss += $v * $v;
            }
            $invNorm = 1.0 / sqrt(($ss / $cols) + $eps);
            for ($j = 0; $j < $cols; $j++) {
                $x->buffer[$offset + $j] = (float)$x->buffer[$offset + $j] * $invNorm * (float)$weight->buffer[$j];
            }
        }
    }

    /**
     * Layer Normalization: (x - mean) / sqrt(var + eps) * gamma + beta
     * In-place on $x.
     */
    public static function layerNormInPlace(Tensor $x, Tensor $gamma, Tensor $beta, float $eps = 1e-5): void
    {
        $cols = $x->shape[count($x->shape) - 1];
        $rows = $x->size / $cols;

        for ($i = 0; $i < $rows; $i++) {
            $offset = $i * $cols;

            // Mean
            $mean = 0.0;
            for ($j = 0; $j < $cols; $j++) $mean += $x->buffer[$offset + $j];
            $mean /= $cols;

            // Variance
            $var = 0.0;
            for ($j = 0; $j < $cols; $j++) {
                $d    = (float)$x->buffer[$offset + $j] - $mean;
                $var += $d * $d;
            }
            $var     /= $cols;
            $invStd   = 1.0 / sqrt($var + $eps);

            for ($j = 0; $j < $cols; $j++) {
                $x->buffer[$offset + $j] =
                    (((float)$x->buffer[$offset + $j] - $mean) * $invStd)
                    * (float)$gamma->buffer[$j]
                    + (float)$beta->buffer[$j];
            }
        }
    }

    /**
     * Batch Normalization (inference mode, no running stats update).
     * In-place.
     */
    public static function batchNormInPlace(Tensor $x, Tensor $gamma, Tensor $beta,
                                             Tensor $runningMean, Tensor $runningVar,
                                             float $eps = 1e-5): void
    {
        $features = $x->shape[count($x->shape) - 1];
        $n        = $x->size / $features;

        for ($i = 0; $i < $n; $i++) {
            $offset = $i * $features;
            for ($j = 0; $j < $features; $j++) {
                $xhat = ((float)$x->buffer[$offset + $j] - (float)$runningMean->buffer[$j])
                        / sqrt((float)$runningVar->buffer[$j] + $eps);
                $x->buffer[$offset + $j] = $xhat * (float)$gamma->buffer[$j] + (float)$beta->buffer[$j];
            }
        }
    }

    // ── Differentiable Normalization ──────────────────────────────────────

    /**
     * Differentiable Root Mean Square Normalization (RMSNorm).
     *
     * Used in LLaMA / Mistral / Gemma as a lightweight alternative to LayerNorm.
     * Applied row-by-row on a 2D input [rows, d]; $w is a learnable scale vector [d].
     *
     * ── Forward (per row t) ──────────────────────────────────────────────
     *
     *   ss_t   = (1/d) · Σ_i x[t,i]²          (mean of squares)
     *   rms_t  = √(ss_t + ε)                   (root mean square + stability)
     *   xhat[t,i] = x[t,i] / rms_t             (normalised value)
     *   y[t,i]    = xhat[t,i] · w[i]           (scale by learnable weight)
     *
     * ── Backward ────────────────────────────────────────────────────────
     *
     *   dL/dw[i] = Σ_t ( dL/dy[t,i] · xhat[t,i] )
     *            = Σ_t ( dL/dy[t,i] · x[t,i] / rms_t )
     *
     *   For x (per row t), let dP[t,i] = dL/dy[t,i] · w[i]  (grad w.r.t. xhat):
     *
     *     DOT_t = Σ_i ( dP[t,i] · x[t,i] )          (dot product in feature space)
     *
     *     dL/dx[t,i] = (1/rms_t) · dP[t,i]
     *                - (x[t,i] / (d · rms_t³)) · DOT_t
     *
     *   Derivation: d(x_k / rms) / dx_i uses the quotient rule with dr/dx_i = x_i/(d·rms),
     *   which yields the cross-term (x_i·DOT)/(d·rms³) subtracted from the diagonal term.
     *
     * @param Tensor $x    [seqLen, d] — 2D input, requiresGrad may be true.
     * @param Tensor $w    [d]         — learnable scale, requiresGrad may be true.
     * @param float  $eps  Stability epsilon (default 1e-6).
     * @return Tensor      [seqLen, d] — normalised + scaled output.
     */
    public static function rmsNorm(Tensor $x, Tensor $w, float $eps = 1e-6): Tensor
    {
        if (count($x->shape) !== 2) {
            throw new \InvalidArgumentException('rmsNorm: $x must be a 2D tensor [rows, d].');
        }
        [$rows, $d] = $x->shape;

        if ($w->size !== $d) {
            throw new \InvalidArgumentException(
                "rmsNorm: weight size {$w->size} does not match last dimension d={$d}."
            );
        }

        $out = Tensor::zeros([$rows, $d]);

        // rmsVals[$t] = rms_t for the backward pass.
        // Storing them avoids recomputing sum(x²) during backward.
        $rmsVals = [];

        for ($t = 0; $t < $rows; $t++) {
            $off = $t * $d;

            // ── Step 1: mean of squares (ss_t) ───────────────────────────
            // Use a PHP loop: BLAS sdot(x, x) also works but adds an FFI cast;
            // the loop is simpler and the d dimension is small (64).
            $ss = 0.0;
            for ($i = 0; $i < $d; $i++) {
                $v   = (float) $x->buffer[$off + $i];
                $ss += $v * $v;
            }
            $rms         = sqrt($ss / $d + $eps);
            $rmsVals[$t] = $rms;

            // ── Step 2: y[t,i] = (x[t,i] / rms_t) * w[i] ───────────────
            for ($i = 0; $i < $d; $i++) {
                $out->buffer[$off + $i] =
                    ((float) $x->buffer[$off + $i] / $rms) * (float) $w->buffer[$i];
            }
        }

        // ── Autograd graph ────────────────────────────────────────────────
        if ($x->requiresGrad || $w->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$x, $w];

            $capturedRms = $rmsVals; // PHP copies the array by value — safe.

            $out->_backward = static function ()
                use ($x, $w, $out, $rows, $d, $capturedRms): void
            {
                // ── dL/dw: accumulate over all rows ──────────────────────
                //
                //   dL/dw[i] += dL/dy[t,i] * x[t,i] / rms_t   for each row t
                //
                if ($w->requiresGrad) {
                    $w->initGrad();
                }

                // ── dL/dx: per-row backward pass ─────────────────────────
                //
                //   dP[t,i] = dL/dy[t,i] * w[i]          (upstream grad w.r.t. xhat)
                //   DOT_t   = Σ_i ( dP[t,i] * x[t,i] )   (feature-space dot product)
                //
                //   dL/dx[t,i] = (1/rms_t) * dP[t,i]
                //              - (x[t,i] / (d * rms_t³)) * DOT_t
                //
                if ($x->requiresGrad) {
                    $x->initGrad();
                }

                for ($t = 0; $t < $rows; $t++) {
                    $off  = $t * $d;
                    $rms  = $capturedRms[$t];
                    $rms3 = $rms * $rms * $rms; // rms_t³ — used in x gradient

                    // ── Pass 1: compute DOT_t and accumulate dL/dw ───────
                    $dot = 0.0;
                    for ($i = 0; $i < $d; $i++) {
                        $dy  = (float) $out->grad[$off + $i];
                        $xi  = (float) $x->buffer[$off + $i];
                        $wi  = (float) $w->buffer[$i];
                        $dPi = $dy * $wi; // dL/dxhat[t,i]

                        $dot += $dPi * $xi;

                        // dL/dw[i] += dL/dy[t,i] * (x[t,i] / rms_t) = dy * xhat[t,i]
                        if ($w->requiresGrad) {
                            $w->grad[$i] = (float) $w->grad[$i]
                                         + $dy * ($xi / $rms);
                        }
                    }

                    // ── Pass 2: compute dL/dx[t,i] for each feature i ────
                    if ($x->requiresGrad) {
                        for ($i = 0; $i < $d; $i++) {
                            $dy  = (float) $out->grad[$off + $i];
                            $xi  = (float) $x->buffer[$off + $i];
                            $wi  = (float) $w->buffer[$i];
                            $dPi = $dy * $wi;

                            // dL/dx[t,i] = (1/rms)*dP_i  -  (x_i / (d*rms³))*DOT
                            $dxi = $dPi / $rms - $xi * $dot / ($d * $rms3);
                            $x->grad[$off + $i] = (float) $x->grad[$off + $i] + $dxi;
                        }
                    }
                }
            };
        }

        return $out;
    }

    // ── Differentiable Slice / Concat (for Multi-Head Attention) ─────────

    /**
     * Slice a contiguous column range from a 2D tensor.
     *
     * Forward:   out[i, j] = x[i, colStart + j]   for j in [0, colEnd − colStart)
     * Backward:  x.grad[i, colStart + j] += out.grad[i, j]
     *
     * Used to split the full Q/K/V matrix [seqLen, dModel] into per-head
     * slices [seqLen, headDim] without any data copies beyond the minimum.
     *
     * @param Tensor $x        2D tensor [rows, cols]
     * @param int    $colStart Inclusive start column (0-based).
     * @param int    $colEnd   Exclusive end column.
     * @return Tensor          [rows, colEnd − colStart]
     */
    public static function sliceCols(Tensor $x, int $colStart, int $colEnd): Tensor
    {
        [$rows, $cols] = $x->shape;
        $headDim = $colEnd - $colStart;

        if ($headDim <= 0 || $colEnd > $cols) {
            throw new \InvalidArgumentException(
                "sliceCols: invalid range [{$colStart}, {$colEnd}) for cols={$cols}."
            );
        }

        $out = new Tensor([$rows, $headDim]);
        $ffi = BlasEngine::get()->ffi;

        // Forward: copy $headDim contiguous floats starting at column $colStart
        // for each row.  cblas_scopy uses incX=1, incY=1 — stride-1 float copy.
        for ($i = 0; $i < $rows; $i++) {
            $srcPtr = \FFI::cast('float*', \FFI::addr($x->buffer[$i * $cols + $colStart]));
            $dstPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $headDim]));
            $ffi->cblas_scopy($headDim, $srcPtr, 1, $dstPtr, 1);
        }

        if ($x->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$x];

            $out->_backward = static function ()
                use ($x, $out, $rows, $cols, $headDim, $colStart, $ffi): void
            {
                $x->initGrad();

                // Scatter out.grad back into the same column range of x.grad.
                // saxpy(n, 1.0, src, 1, dst, 1) ≡ dst += src (element-wise).
                for ($i = 0; $i < $rows; $i++) {
                    $gSrc = \FFI::cast('float*', \FFI::addr($out->grad[$i * $headDim]));
                    $gDst = \FFI::cast('float*', \FFI::addr($x->grad[$i * $cols + $colStart]));
                    $ffi->cblas_saxpy($headDim, 1.0, $gSrc, 1, $gDst, 1);
                }
            };
        }

        return $out;
    }

    /**
     * Concatenate 2D tensors along the column axis.
     *
     * Forward:   out[i, offset_k + j] = tensors[k][i, j]
     *            where offset_k = Σ_{m<k} tensors[m]->shape[1]
     * Backward:  tensors[k].grad[i, j] += out.grad[i, offset_k + j]
     *
     * Used to merge per-head attention outputs [seqLen, headDim] × nHeads
     * back into [seqLen, dModel] before the final Wo projection.
     *
     * @param Tensor[] $tensors  All must be 2D with the same row count.
     * @return Tensor            [rows, Σ cols_k]
     */
    public static function concatCols(array $tensors): Tensor
    {
        if (empty($tensors)) {
            throw new \InvalidArgumentException('concatCols: tensor array must not be empty.');
        }

        $rows      = $tensors[0]->shape[0];
        $totalCols = 0;
        $colOffsets = []; // colOffsets[$k] = start column of tensors[$k] in the output

        foreach ($tensors as $k => $t) {
            if (count($t->shape) !== 2) {
                throw new \InvalidArgumentException("concatCols: tensor[{$k}] must be 2D.");
            }
            if ($t->shape[0] !== $rows) {
                throw new \InvalidArgumentException(
                    "concatCols: tensor[{$k}] has {$t->shape[0]} rows; expected {$rows}."
                );
            }
            $colOffsets[] = $totalCols;
            $totalCols   += $t->shape[1];
        }

        $out = new Tensor([$rows, $totalCols]);
        $ffi = BlasEngine::get()->ffi;
        $requiresGrad = false;

        // Forward: copy each tensor's data into its column range in $out
        foreach ($tensors as $k => $t) {
            $headDim = $t->shape[1];
            $colOff  = $colOffsets[$k];

            for ($i = 0; $i < $rows; $i++) {
                $srcPtr = \FFI::cast('float*', \FFI::addr($t->buffer[$i * $headDim]));
                $dstPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $totalCols + $colOff]));
                $ffi->cblas_scopy($headDim, $srcPtr, 1, $dstPtr, 1);
            }

            if ($t->requiresGrad) {
                $requiresGrad = true;
            }
        }

        if ($requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = $tensors;

            $capturedOffsets = $colOffsets; // copied by value

            $out->_backward = static function ()
                use ($tensors, $out, $rows, $totalCols, $capturedOffsets, $ffi): void
            {
                // Split out.grad back into each input tensor's grad.
                foreach ($tensors as $k => $t) {
                    if (!$t->requiresGrad) {
                        continue;
                    }
                    $t->initGrad();

                    $headDim = $t->shape[1];
                    $colOff  = $capturedOffsets[$k];

                    for ($i = 0; $i < $rows; $i++) {
                        $gSrc = \FFI::cast('float*', \FFI::addr($out->grad[$i * $totalCols + $colOff]));
                        $gDst = \FFI::cast('float*', \FFI::addr($t->grad[$i * $headDim]));
                        $ffi->cblas_saxpy($headDim, 1.0, $gSrc, 1, $gDst, 1);
                    }
                }
            };
        }

        return $out;
    }

    // ── Loss Functions ────────────────────────────────────────────────────

    /**
     * Cross-Entropy Loss (averaged).
     * $logits: [N, C] raw scores (pre-softmax)
     * $targets: [N] integer class indices
     */
    public static function crossEntropyLoss(Tensor $logits, array $targets): float
    {
        $n    = $logits->shape[0];
        $C    = $logits->shape[1];
        $loss = 0.0;

        for ($i = 0; $i < $n; $i++) {
            $offset = $i * $C;
            $maxVal = -INF;
            for ($j = 0; $j < $C; $j++) {
                if ($logits->buffer[$offset + $j] > $maxVal) $maxVal = $logits->buffer[$offset + $j];
            }
            $logSum = 0.0;
            for ($j = 0; $j < $C; $j++) {
                $logSum += exp($logits->buffer[$offset + $j] - $maxVal);
            }
            $logSum  = log($logSum) + $maxVal;
            $loss   += $logSum - (float)$logits->buffer[$offset + $targets[$i]];
        }
        return $loss / $n;
    }

    /**
     * Mean Squared Error Loss.
     * $pred, $target: same shape.
     */
    public static function mseLoss(Tensor $pred, Tensor $target): float
    {
        if ($pred->size !== $target->size) throw new \InvalidArgumentException('MSE: size mismatch.');
        $sum = 0.0;
        for ($i = 0; $i < $pred->size; $i++) {
            $d    = (float)$pred->buffer[$i] - (float)$target->buffer[$i];
            $sum += $d * $d;
        }
        return $sum / $pred->size;
    }

    // ── LAPACK: Linear Algebra ─────────────────────────────────────────────

    /**
     * LU factorization + matrix inverse using LAPACK sgetrf + sgetri.
     * $A must be square [n, n]. Returns inverse as a new Tensor.
     */
    public static function inverse(Tensor $A): Tensor
    {
        if (count($A->shape) !== 2 || $A->shape[0] !== $A->shape[1]) {
            throw new \InvalidArgumentException('inverse() requires a square 2D tensor.');
        }
        $n    = $A->shape[0];
        $copy = $A->clone();
        $ipiv = BlasEngine::get()->allocInt($n);
        $ffi  = BlasEngine::get()->lapacke;

        $info = $ffi->LAPACKE_sgetrf(BlasEngine::LAPACK_ROW_MAJOR, $n, $n, $copy->buffer, $n, $ipiv);
        if ($info !== 0) throw new \RuntimeException("sgetrf failed with info={$info}.");

        $info = $ffi->LAPACKE_sgetri(BlasEngine::LAPACK_ROW_MAJOR, $n, $copy->buffer, $n, $ipiv);
        if ($info !== 0) throw new \RuntimeException("sgetri failed with info={$info}.");

        return $copy;
    }

    /**
     * Solve the linear system A * X = B using LAPACK sgetrs.
     * $A: [n, n], $B: [n, nrhs] → returns X: [n, nrhs]
     */
    public static function solve(Tensor $A, Tensor $B): Tensor
    {
        $n    = $A->shape[0];
        $nrhs = count($B->shape) === 2 ? $B->shape[1] : 1;
        $Ac   = $A->clone();
        $X    = $B->clone();
        $ipiv = BlasEngine::get()->allocInt($n);
        $ffi  = BlasEngine::get()->lapacke;

        $info = $ffi->LAPACKE_sgetrf(BlasEngine::LAPACK_ROW_MAJOR, $n, $n, $Ac->buffer, $n, $ipiv);
        if ($info !== 0) throw new \RuntimeException("solve/sgetrf failed with info={$info}.");

        $info = $ffi->LAPACKE_sgetrs(BlasEngine::LAPACK_ROW_MAJOR, 'N', $n, $nrhs, $Ac->buffer, $n, $ipiv, $X->buffer, $nrhs);
        if ($info !== 0) throw new \RuntimeException("solve/sgetrs failed with info={$info}.");

        return $X;
    }

    /**
     * Thin SVD of A. Returns [$U, $s, $Vt].
     * $A: [m, n] → U: [m, k], s: [k], Vt: [k, n]  where k = min(m,n)
     */
    public static function svd(Tensor $A): array
    {
        [$m, $n] = $A->shape;
        $k    = min($m, $n);
        $Ac   = $A->clone();
        $U    = new Tensor([$m, $k]);
        $s    = new Tensor([$k]);
        $Vt   = new Tensor([$k, $n]);
        $superb = new Tensor([$k - 1]);
        $ffi  = BlasEngine::get()->lapacke;

        $info = $ffi->LAPACKE_sgesvd(
            BlasEngine::LAPACK_ROW_MAJOR, 'S', 'S',
            $m, $n, $Ac->buffer, $n,
            $s->buffer,
            $U->buffer, $k,
            $Vt->buffer, $n,
            $superb->buffer
        );
        if ($info !== 0) throw new \RuntimeException("SVD failed with info={$info}.");

        return [$U, $s, $Vt];
    }

    /**
     * Symmetric eigendecomposition using LAPACK ssyev.
     * $A: [n, n] symmetric → [eigenvalues: [n], eigenvectors: [n, n]]
     */
    public static function eigh(Tensor $A): array
    {
        if ($A->shape[0] !== $A->shape[1]) throw new \InvalidArgumentException('eigh: must be square.');
        $n  = $A->shape[0];
        $Ac = $A->clone();
        $w  = new Tensor([$n]);
        $ffi = BlasEngine::get()->lapacke;

        $info = $ffi->LAPACKE_ssyev(BlasEngine::LAPACK_ROW_MAJOR, 'V', 'U', $n, $Ac->buffer, $n, $w->buffer);
        if ($info !== 0) throw new \RuntimeException("eigh/ssyev failed with info={$info}.");

        return [$w, $Ac]; // eigenvalues, eigenvectors (columns of Ac)
    }

    /**
     * Cholesky solve: A * X = B where A is symmetric positive-definite.
     * $A: [n, n], $B: [n, nrhs]
     */
    public static function choleskySolve(Tensor $A, Tensor $B): Tensor
    {
        $n    = $A->shape[0];
        $nrhs = count($B->shape) === 2 ? $B->shape[1] : 1;
        $Ac   = $A->clone();
        $X    = $B->clone();
        $ffi  = BlasEngine::get()->lapacke;

        $info = $ffi->LAPACKE_sposv(BlasEngine::LAPACK_ROW_MAJOR, 'U', $n, $nrhs, $Ac->buffer, $n, $X->buffer, $nrhs);
        if ($info !== 0) throw new \RuntimeException("choleskySolve/sposv failed with info={$info}.");

        return $X;
    }

    /**
     * Least-squares solve: min ||A*X - B||₂ using LAPACK sgels.
     */
    public static function lstsq(Tensor $A, Tensor $B): Tensor
    {
        [$m, $n] = $A->shape;
        $nrhs  = count($B->shape) === 2 ? $B->shape[1] : 1;
        $Ac    = $A->clone();
        $X     = $B->clone();
        $ffi   = BlasEngine::get()->lapacke;

        $info = $ffi->LAPACKE_sgels(BlasEngine::LAPACK_ROW_MAJOR, 'N', $m, $n, $nrhs, $Ac->buffer, $n, $X->buffer, $nrhs);
        if ($info !== 0) throw new \RuntimeException("lstsq/sgels failed with info={$info}.");

        return $X;
    }

    /**
     * Named alias for lstsq() — exposed under the scikit-learn-familiar name.
     *
     * Solves the linear least-squares problem  min ||A*x − B||₂  using
     * LAPACKE_sgels (QR / LQ factorisation).
     *
     * @param Tensor $A  [m, n] coefficient matrix
     * @param Tensor $B  [m] or [m, nrhs] right-hand side(s)
     * @return Tensor    Solution x in the first n rows of the returned buffer.
     *                   Subsequent rows contain residual information.
     */
    public static function leastSquares(Tensor $A, Tensor $B): Tensor
    {
        return self::lstsq($A, $B);
    }

    // ── Embeddings / Indexing ─────────────────────────────────────────────

    /**
     * Embedding lookup: given a weight matrix [vocab_size, d_model] and
     * an array of token IDs, returns [seq_len, d_model] via BLAS scopy.
     */
    public static function embedding(Tensor $weight, array $tokenIds): Tensor
    {
        $dModel  = $weight->shape[1];
        $seqLen  = count($tokenIds);
        $out     = new Tensor([$seqLen, $dModel]);
        $ffi     = BlasEngine::get()->ffi;

        for ($i = 0; $i < $seqLen; $i++) {
            $id  = $tokenIds[$i];
            $src = \FFI::cast('float*', \FFI::addr($weight->buffer[$id * $dModel]));
            $dst = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $dModel]));
            $ffi->cblas_scopy($dModel, $src, 1, $dst, 1);
        }

        return $out;
    }

    /**
     * Add bias (1D) to each row of a 2D tensor in-place.
     * Uses BLAS saxpy per row.
     */
    public static function addBiasInPlace(Tensor $x, Tensor $bias): void
    {
        $cols = $x->shape[count($x->shape) - 1];
        $rows = $x->size / $cols;
        $ffi  = BlasEngine::get()->ffi;

        for ($i = 0; $i < $rows; $i++) {
            $dst = \FFI::cast('float*', \FFI::addr($x->buffer[$i * $cols]));
            $ffi->cblas_saxpy($cols, 1.0, $bias->buffer, 1, $dst, 1);
        }
    }

    /**
     * Concatenate tensors along axis 0 (row stacking for 2D tensors).
     */
    public static function concat(Tensor ...$tensors): Tensor
    {
        if (empty($tensors)) throw new \InvalidArgumentException('concat requires at least one tensor.');

        $cols   = $tensors[0]->shape[count($tensors[0]->shape) - 1];
        $totalRows = 0;
        foreach ($tensors as $t) $totalRows += (int)($t->size / $cols);

        $out    = new Tensor([$totalRows, $cols]);
        $offset = 0;
        $ffi    = BlasEngine::get()->ffi;

        foreach ($tensors as $t) {
            $dst = \FFI::cast('float*', \FFI::addr($out->buffer[$offset]));
            $ffi->cblas_scopy($t->size, $t->buffer, 1, $dst, 1);
            $offset += $t->size;
        }

        return $out;
    }

    /**
     * Apply a causal (lower-triangular) mask to attention scores in-place.
     * Sets upper-triangular elements to -INF so softmax assigns them zero prob.
     * $scores: [seq_len, seq_len]
     */
    public static function applyCausalMaskInPlace(Tensor $scores): void
    {
        $seq = $scores->shape[0];
        for ($i = 0; $i < $seq; $i++) {
            for ($j = $i + 1; $j < $seq; $j++) {
                $scores->buffer[$i * $seq + $j] = -INF;
            }
        }
    }

    // ── Private Helpers ───────────────────────────────────────────────────

    private static function softmax1DInPlace(\FFI\CData $buf, int $offset, int $len): void
    {
        $maxVal = -INF;
        for ($j = 0; $j < $len; $j++) {
            if ($buf[$offset + $j] > $maxVal) $maxVal = $buf[$offset + $j];
        }
        $sum = 0.0;
        for ($j = 0; $j < $len; $j++) {
            $v = exp((float)$buf[$offset + $j] - $maxVal);
            $buf[$offset + $j] = $v;
            $sum += $v;
        }
        $invSum = 1.0 / $sum;
        for ($j = 0; $j < $len; $j++) {
            $buf[$offset + $j] = (float)$buf[$offset + $j] * $invSum;
        }
    }
}