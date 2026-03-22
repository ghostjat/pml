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
        // Resolve effective transpose flags (Tensor::T() sets _transposed)
        $effTransA = $transA || $A->_transposed;
        $effTransB = $transB || $B->_transposed;

        // Effective shapes after transpose
        [$Arows, $Acols] = $A->_transposed
            ? [$A->shape[1], $A->shape[0]]
            : [$A->shape[0], $A->shape[1]];
        [$Brows, $Bcols] = $B->_transposed
            ? [$B->shape[1], $B->shape[0]]
            : [$B->shape[0], $B->shape[1]];

        $M = $effTransA ? $Acols : $Arows;
        $K = $effTransA ? $Arows : $Acols;
        $N = $effTransB ? $Brows : $Bcols;
        $K2 = $effTransB ? $Bcols : $Brows;

        if ($K !== $K2) {
            throw new \InvalidArgumentException(
                "matmul: inner dimensions mismatch. op(A) has K={$K}, op(B) has K={$K2}."
            );
        }

        $out = $C ?? new Tensor([$M, $N]);

        // Leading dimensions are always the PHYSICAL column count
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
     */
    public static function add(Tensor $A, Tensor $B): Tensor
    {
        $C = $A->clone();
        self::saxpy($B, $C, 1.0);
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