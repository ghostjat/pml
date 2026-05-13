<?php
declare(strict_types=1);

namespace Pml;
use Pml\Lib\TensorEngine;

/**
 * The PHP PyTorch/NumPy equivalent API.
 *
 * Optimized for JIT:
 * - Cached FFI instance avoids repeated TensorEngine::get() calls.
 * - Reflection for `wrap()` is cached (already done, kept).
 * - dtype branching uses `match` (PHP 8.0+) instead of if/elseif chains.
 * - `shape()` uses `array_slice` on FFI array view for O(1) conversion.
 * - `__destruct` uses cached FFI.
 * - All methods type‑hinted; class is final to assist devirtualization.
 */
final class Tensor {
    // Data Type Enums mapping to the C definitions
    public const DTYPE_FLOAT32 = 0;
    public const DTYPE_INT32 = 1;
    public const DTYPE_INT64 = 2;

    // Cached FFI instance – loaded once per request.
    private static ?\FFI $ffi = null;

    public ?\FFI\CData $ptr;
    private ?\FFI\CData $buffer = null;
    private bool $owned = false;
    
    // Crucial: Holds a reference to the parent tensor if this is a zero-copy view
    private ?self $parent = null; 

    /**
     * @param array $shape e.g. [10, 10]
     * @param int $dtype defaults to Tensor::DTYPE_FLOAT32
     * @param \FFI\CData|null $arena Optional memory pool arena pointer for O(1) allocation
     */
    public function __construct(array $shape = [], int $dtype = self::DTYPE_FLOAT32, ?\FFI\CData $arena = null) 
    {
        // Allow empty constructor for factory methods like fromArray()
        if (empty($shape)) {
            return;
        }

        $ndim = count($shape);
        if ($ndim > 8) {
            throw new \InvalidArgumentException("Tensor dimensions must be 1 to 8.");
        }

        $ffi = self::ffi();
        $cShape = $ffi->new("int[$ndim]");
        
        foreach ($shape as $i => $dim) {
            $cShape[$i] = $dim;
        }

        $this->ptr = $arena !== null
            ? $ffi->tensor_create_arena($ndim, $ffi->cast("int*", $cShape), $dtype, $arena)
            : $ffi->tensor_create_dtype($ndim, $ffi->cast("int*", $cShape), $dtype);
        
        self::checkError();
        // Arena tensors: both the struct and its data live inside the arena
        // memory block.  arena_destroy() bulk-frees everything at once, so
        // the PHP destructor must NOT call tensor_free() — that would be a
        // use-after-free on already-freed arena memory.
        $this->owned = ($arena === null);

        // Use match for faster dtype dispatch
        $this->buffer = match ($dtype) {
            self::DTYPE_INT32 => $ffi->cast("int32_t*", $this->ptr->data),
            self::DTYPE_INT64 => $ffi->cast("int64_t*", $this->ptr->data),
            default           => $ffi->cast("float*", $this->ptr->data),
        };
    }

    /** @var \ReflectionClass<self>|null Cached once — avoids one object alloc per tensor output. */
    private static ?\ReflectionClass $_reflector = null;

    /**
     * Wrap a C TensorC* into a PHP Tensor object.
     */
    public static function wrap(?\FFI\CData $ptr, ?self $parent = null): self {
        if ($ptr === null) {
            self::checkError();
            throw new \RuntimeException("C-Engine returned NULL without setting error.");
        }

        if (self::$_reflector === null) {
            self::$_reflector = new \ReflectionClass(self::class);
        }
        $t = self::$_reflector->newInstanceWithoutConstructor();
        $t->ptr = $ptr;
        
        $ffi = self::ffi();
        $dtype = $ptr->dtype;
        
        $t->buffer = match ($dtype) {
            self::DTYPE_INT32 => $ffi->cast("int32_t*", $ptr->data),
            self::DTYPE_INT64 => $ffi->cast("int64_t*", $ptr->data),
            default           => $ffi->cast("float*", $ptr->data),
        };
        
        $t->owned = true; 
        $t->parent = $parent; 
        return $t;
    }

    /**
     * Returns the cached FFI instance (lazy initialised).
     */
    private static function ffi(): \FFI {
        return self::$ffi ??= TensorEngine::get();
    }

    /**
     * Translates a C-engine error into a PHP RuntimeException gracefully.
     */
    private static function checkError(): void {
        $ffi = self::ffi();
        if ($ffi->tensor_check_error()) {
            // Clear BEFORE reading the string so the error never leaks into
            // subsequent tests even if FFI::string() itself throws.
            $errPtr = $ffi->tensor_get_last_error();
            // PHP FFI may return const char* as either FFI\CData or a plain PHP
            // string depending on the runtime version.
            $err = $errPtr instanceof \FFI\CData ? \FFI::string($errPtr) : (string)$errPtr;
            $ffi->tensor_clear_error();
            throw new \RuntimeException("C-Engine Error: {$err}");
            
        }
    }

    // --- METADATA GETTERS ---
    public function dtype(): int { return $this->ptr->dtype; }
    public function shape(): array {
        $n   = (int)$this->ptr->ndim;
        $out = [];
        for ($i = 0; $i < $n; $i++) {
            $out[] = (int)$this->ptr->shape[$i];
        }
        return $out;
    }
    public function size(): int { return $this->ptr->total_size; }
    public function ndim(): int { return $this->ptr->ndim; }

    // --- ZERO-COPY VIEWS & SLICING ---
    public function view(): self { 
        $res = self::wrap(self::ffi()->tensor_view($this->ptr), $this); 
        self::checkError(); return $res;
    }
    public function slice(int $axis, int $start, int $length): self { 
        $res = self::wrap(self::ffi()->tensor_slice($this->ptr, $axis, $start, $length), $this); 
        self::checkError(); return $res;
    }
    public function sliceStep(int $axis, int $start, int $end, int $step): self { 
        $res = self::wrap(self::ffi()->tensor_slice_step($this->ptr, $axis, $start, $end, $step), $this); 
        self::checkError(); return $res;
    }
    public function row(int $row): self { 
        $res = self::wrap(self::ffi()->tensor_row_view($this->ptr, $row), $this); 
        self::checkError(); return $res;
    }
    public function col(int $col): self { 
        $res = self::wrap(self::ffi()->tensor_column_view($this->ptr, $col), $this); 
        self::checkError(); return $res;
    }

    // --- INITIALIZERS & CREATION ---
    public function fill(float $val): self { 
        self::ffi()->tensor_fill($this->ptr, $val); 
        self::checkError(); return $this; 
    }
    public function copy(): self { 
        $res = self::wrap(self::ffi()->tensor_copy($this->ptr)); 
        self::checkError(); return $res;
    }
    public function isContiguous(): bool {
        $res = self::ffi()->tensor_is_contiguous($this->ptr);
        self::checkError(); return $res;
    }
    /**
     * Return a contiguous copy of this tensor, or $this if already contiguous.
     * Zero-cost for the common case; one C memcpy for non-contiguous views.
     */
    public function contiguous(): self {
        return $this->isContiguous() ? $this : $this->copy();
    }

    /**
     * Create an uninitialized tensor with the same shape and dtype as $t.
     * Skips zero-fill — use when every element will be overwritten immediately.
     */
    public static function emptyLike(self $t): self {
        $ffi  = self::ffi();
        $shape = $t->shape();
        $ndim  = count($shape);
        $cShape = $ffi->new("int[$ndim]");
        foreach ($shape as $i => $d) $cShape[$i] = $d;
        $res = self::wrap($ffi->tensor_create_uninitialized($ndim, $ffi->cast("int*", $cShape), $t->dtype()));
        self::checkError(); return $res;
    }
    
    public static function zeros(int ...$shape): self {
        $ndim = count($shape); $ffi = self::ffi();
        $cShape = $ffi->new("int[$ndim]"); foreach ($shape as $i => $dim) $cShape[$i] = $dim;
        $res = self::wrap($ffi->tensor_zeros($ndim, $ffi->cast("int*", $cShape)));
        self::checkError(); return $res;
    }

    public static function ones(int ...$shape): self {
        $ndim = count($shape); $ffi = self::ffi();
        $cShape = $ffi->new("int[$ndim]"); foreach ($shape as $i => $dim) $cShape[$i] = $dim;
        $res = self::wrap($ffi->tensor_ones($ndim, $ffi->cast("int*", $cShape)));
        self::checkError(); return $res;
    }

    public static function zerosArena(\FFI\CData $arena, int ...$shape): self {
        $t = new self($shape, self::DTYPE_FLOAT32, $arena);
        self::ffi()->tensor_fill($t->ptr, 0.0);
        self::checkError();
        return $t;
    }

    public static function onesArena(\FFI\CData $arena, int ...$shape): self {
        $t = new self($shape, self::DTYPE_FLOAT32, $arena);
        self::ffi()->tensor_fill($t->ptr, 1.0);
        self::checkError();
        return $t;
    }

    public static function range(float $start, float $end, float $step = 1.0): self { 
        $res = self::wrap(self::ffi()->tensor_range($start, $end, $step)); 
        self::checkError(); return $res;
    }
    public static function linspace(float $start, float $end, int $steps): self { 
        $res = self::wrap(self::ffi()->tensor_linspace($start, $end, $steps)); 
        self::checkError(); return $res;
    }

    public static function randomNormal(array $shape, float $mean = 0.0, float $stddev = 1.0, ?\FFI\CData $arena = null): self {
        $t = new self($shape, self::DTYPE_FLOAT32, $arena); 
        self::ffi()->tensor_random_normal($t->ptr, $mean, $stddev); 
        self::checkError(); return $t;
    }
    public static function randomUniform(array $shape, float $minVal = 0.0, float $maxVal = 1.0, ?\FFI\CData $arena = null): self {
        $t = new self($shape, self::DTYPE_FLOAT32, $arena); 
        self::ffi()->tensor_random_uniform($t->ptr, $minVal, $maxVal); 
        self::checkError(); return $t;
    }
    public function randomChoice(int $n, bool $replace = true): self { 
        $res = self::wrap(self::ffi()->tensor_random_choice($this->ptr, $n, $replace)); 
        self::checkError(); return $res;
    }
    public function randomPermutation(): self { 
        $res = self::wrap(self::ffi()->tensor_random_permutation($this->ptr)); 
        self::checkError(); return $res;
    }

    // --- FUSED NEURAL NETWORK KERNELS ---

    /**
     * Fused fully-connected layer: out = X @ W^T + bias
     *
     * Single SGEMM + one AVX2 bias pass — no temporary tensor between them.
     * Replaces: $x->matmul($w->transpose())->addInplace($bias)
     *
     * X   : [m, k]  (or any shape where the last dim is k)
     * W   : [n, k]  (weight matrix, rows = output neurons)
     * bias: [n]     (optional)
     * → out: [m, n]
     */
    public function linear(self $W, ?self $bias = null): self {
        $res = self::wrap(self::ffi()->tensor_linear($this->ptr, $W->ptr, $bias?->ptr));
        self::checkError();
        return $res;
    }

    /**
     * Fused add + ReLU: out = relu(A + B)
     * Saves one full output-buffer allocation vs add() + relu().
     */
    public function addRelu(self $b): self {
        $res = self::wrap(self::ffi()->tensor_add_relu($this->ptr, $b->ptr));
        self::checkError();
        return $res;
    }

    /**
     * Fused multiply-add (FMA): out = $this * B + C
     * Uses _mm256_fmadd_ps — one instruction, one memory pass.
     */
    public function mulAdd(self $B, self $C): self {
        $res = self::wrap(self::ffi()->tensor_mul_add($this->ptr, $B->ptr, $C->ptr));
        self::checkError();
        return $res;
    }

    /**
     * Configure OpenMP and BLAS thread counts independently.
     *
     * Call once at startup to prevent oversubscription:
     *   Tensor::configureThreading(cores: 8, blasThreads: 1)   // outer-OMP workloads
     *   Tensor::configureThreading(cores: 8, blasThreads: 8)   // pure-BLAS workloads
     */
    public static function configureThreading(int $ompThreads, int $blasThreads = 1): void {
        self::ffi()->tensor_configure_threading($ompThreads, $blasThreads);
        self::checkError();
    }

    // --- LEGACY FUSED KERNELS & HARDWARE INFERENCE ---

    public static function fusedBceLossAndGrad(self $preds, self $targets, ?self $grads = null): float {
        $ffi = self::ffi();
        $outLoss = $ffi->new("float[1]"); 
        $gradsPtr = $grads?->ptr;
        
        $ffi->tensor_fused_bce_loss_and_grad($preds->ptr, $targets->ptr, $gradsPtr, \FFI::addr($outLoss[0]));
        self::checkError();
        return $outLoss[0];
    }

    public static function fusedAdamStep(self $param, self $grad, self $m, self $v, float $lr, float $b1, float $b2, float $eps, int $t): void {
        self::ffi()->tensor_fused_adam_step($param->ptr, $grad->ptr, $m->ptr, $v->ptr, $lr, $b1, $b2, $eps, $t);
        self::checkError();
    }

    public static function fusedSgdStep(self $param, self $grad, float $lr): void {
        self::ffi()->tensor_fused_sgd_step($param->ptr, $grad->ptr, $lr);
        self::checkError();
    }

    public static function fusedRmsPropStep(self $param, self $grad, self $cache, float $lr, float $decay, float $eps): void {
        self::ffi()->tensor_fused_rmsprop_step($param->ptr, $grad->ptr, $cache->ptr, $lr, $decay, $eps);
        self::checkError();
    }

    public static function fusedAdaGradStep(self $param, self $grad, self $acc, float $lr, float $eps): void {
        self::ffi()->tensor_fused_adagrad_step($param->ptr, $grad->ptr, $acc->ptr, $lr, $eps);
        self::checkError();
    }

    public static function fusedAdamWStep(self $param, self $grad, self $m, self $v, float $lr, float $b1, float $b2, float $eps, int $t, float $wd): void {
        self::ffi()->tensor_fused_adamw_step($param->ptr, $grad->ptr, $m->ptr, $v->ptr, $lr, $b1, $b2, $eps, $t, $wd);
        self::checkError();
    }

    // --- SHAPE MUTATIONS ---
    public function reshape(int ...$newShape): self {
        $ndim = count($newShape); $ffi = self::ffi();
        $cShape = $ffi->new("int[$ndim]"); foreach ($newShape as $i => $dim) $cShape[$i] = $dim;
        $res = self::wrap($ffi->tensor_reshape($this->ptr, $ndim, $ffi->cast("int*", $cShape)), $this);
        self::checkError(); return $res;
    }
    public function flatten(): self { 
        $res = self::wrap(self::ffi()->tensor_flatten($this->ptr), $this); 
        self::checkError(); return $res;
    }
    public function expandDims(int $axis): self { 
        $res = self::wrap(self::ffi()->tensor_expand_dims($this->ptr, $axis), $this); 
        self::checkError(); return $res;
    }
    public function squeeze(): self { 
        $res = self::wrap(self::ffi()->tensor_squeeze($this->ptr), $this); 
        self::checkError(); return $res;
    }
    public function transpose(): self { 
        $res = self::wrap(self::ffi()->tensor_transpose_2d($this->ptr), $this); 
        self::checkError(); return $res;
    }
    
    public function transposeNd(array $axes): self {
        $ffi = self::ffi();
        $cAxes = $ffi->new("int[" . count($axes) . "]"); foreach ($axes as $i => $ax) $cAxes[$i] = $ax;
        $res = self::wrap($ffi->tensor_transpose_nd($this->ptr, $ffi->cast("int*", $cAxes)), $this);
        self::checkError(); return $res;
    }
    public function swapaxes(int $axis1, int $axis2): self { 
        $res = self::wrap(self::ffi()->tensor_swapaxes($this->ptr, $axis1, $axis2), $this); 
        self::checkError(); return $res;
    }

    // --- MATH BINARY & SCALAR ---
    public function add(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_add($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function sub(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_sub($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function mul(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_mul($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function div(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_div($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function addScalar(float $val): self { $res = self::wrap(self::ffi()->tensor_add_scalar($this->ptr, $val)); self::checkError(); return $res; }
    public function mulScalar(float $val): self { $res = self::wrap(self::ffi()->tensor_mul_scalar($this->ptr, $val)); self::checkError(); return $res; }
    public function pow(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_pow($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function clip(float $min, float $max): self { $res = self::wrap(self::ffi()->tensor_clip($this->ptr, $min, $max)); self::checkError(); return $res; }

    // O(1) FFI Logic
    public function lessScalar(float $val): self { $res = self::wrap(self::ffi()->tensor_less_scalar_f32($this->ptr, $val)); self::checkError(); return $res; }
    public function greaterScalar(float $val): self { $res = self::wrap(self::ffi()->tensor_greater_scalar_f32($this->ptr, $val)); self::checkError(); return $res; }

    // --- IN-PLACE OPS ---
    public function addInplace(Tensor $b): self { self::ffi()->tensor_add_inplace($this->ptr, $b->ptr); self::checkError(); return $this; }
    public function subInplace(Tensor $b): self { self::ffi()->tensor_sub_inplace($this->ptr, $b->ptr); self::checkError(); return $this; }
    public function mulInplace(Tensor $b): self { self::ffi()->tensor_mul_inplace($this->ptr, $b->ptr); self::checkError(); return $this; }
    public function divInplace(Tensor $b): self { self::ffi()->tensor_div_inplace($this->ptr, $b->ptr); self::checkError(); return $this; }
    public function addScalarInplace(float $val): self { self::ffi()->tensor_add_scalar_inplace($this->ptr, $val); self::checkError(); return $this; }
    public function mulScalarInplace(float $val): self { self::ffi()->tensor_mul_scalar_inplace($this->ptr, $val); self::checkError(); return $this; }
    public function clampInplace(float $lo, float $hi): self { self::ffi()->tensor_clamp_inplace($this->ptr, $lo, $hi); self::checkError(); return $this; }

    // --- MATH UNARY ---
    public function sqrt(): self { $res = self::wrap(self::ffi()->tensor_sqrt($this->ptr)); self::checkError(); return $res; }
    public function square(): self { $res = self::wrap(self::ffi()->tensor_square($this->ptr)); self::checkError(); return $res; }
    public function abs(): self { $res = self::wrap(self::ffi()->tensor_abs($this->ptr)); self::checkError(); return $res; }
    public function sign(): self { $res = self::wrap(self::ffi()->tensor_sign($this->ptr)); self::checkError(); return $res; }
    public function exp(): self { $res = self::wrap(self::ffi()->tensor_exp($this->ptr)); self::checkError(); return $res; }
    public function log(): self { $res = self::wrap(self::ffi()->tensor_log($this->ptr)); self::checkError(); return $res; }
    public function log1p(): self { $res = self::wrap(self::ffi()->tensor_log1p($this->ptr)); self::checkError(); return $res; }
    public function round(): self { $res = self::wrap(self::ffi()->tensor_round($this->ptr)); self::checkError(); return $res; }
    public function floor(): self { $res = self::wrap(self::ffi()->tensor_floor($this->ptr)); self::checkError(); return $res; }
    public function ceil(): self { $res = self::wrap(self::ffi()->tensor_ceil($this->ptr)); self::checkError(); return $res; }
    public function sigmoid(): self { $res = self::wrap(self::ffi()->tensor_sigmoid($this->ptr)); self::checkError(); return $res; }
    public function tanh(): self { $res = self::wrap(self::ffi()->tensor_tanh($this->ptr)); self::checkError(); return $res; }
    public function relu(): self { $res = self::wrap(self::ffi()->tensor_relu($this->ptr)); self::checkError(); return $res; }
    public function sin(): self { $res = self::wrap(self::ffi()->tensor_sin($this->ptr)); self::checkError(); return $res; }
    public function cos(): self { $res = self::wrap(self::ffi()->tensor_cos($this->ptr)); self::checkError(); return $res; }
    public function tan(): self { $res = self::wrap(self::ffi()->tensor_tan($this->ptr)); self::checkError(); return $res; }
    public function asin(): self { $res = self::wrap(self::ffi()->tensor_asin($this->ptr)); self::checkError(); return $res; }
    public function acos(): self { $res = self::wrap(self::ffi()->tensor_acos($this->ptr)); self::checkError(); return $res; }
    public function atan(): self { $res = self::wrap(self::ffi()->tensor_atan($this->ptr)); self::checkError(); return $res; }

    // --- LOGICAL & MESSY DATA ---
    public function equal(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_equal($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function notEqual(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_not_equal($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function greater(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_greater($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function greaterEqual(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_greater_equal($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function less(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_less($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function lessEqual(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_less_equal($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function logicalNot(): self { $res = self::wrap(self::ffi()->tensor_logical_not($this->ptr)); self::checkError(); return $res; }
    
    public function isNan(): self { $res = self::wrap(self::ffi()->tensor_isnan($this->ptr)); self::checkError(); return $res; }
    public function isInf(): self { $res = self::wrap(self::ffi()->tensor_isinf($this->ptr)); self::checkError(); return $res; }
    public function nanToNumInplace(float $nan_val = 0.0, float $posinf = 1e38, float $neginf = -1e38): self { self::ffi()->tensor_nan_to_num_inplace($this->ptr, $nan_val, $posinf, $neginf); self::checkError(); return $this; }
    public function any(): bool { $res = self::ffi()->tensor_any($this->ptr); self::checkError(); return $res; }
    public function all(): bool { $res = self::ffi()->tensor_all($this->ptr); self::checkError(); return $res; }

    // --- FANCY INDEXING & SETS ---
    public function where(Tensor $x, Tensor $y): self { $res = self::wrap(self::ffi()->tensor_where($this->ptr, $x->ptr, $y->ptr)); self::checkError(); return $res; }
    public function booleanIndex(Tensor $mask): self { $res = self::wrap(self::ffi()->tensor_boolean_index($this->ptr, $mask->ptr)); self::checkError(); return $res; }
    public function take(Tensor $indices, int $axis): self { $res = self::wrap(self::ffi()->tensor_take($this->ptr, $indices->ptr, $axis)); self::checkError(); return $res; }
    public function unique(): self { $res = self::wrap(self::ffi()->tensor_unique($this->ptr)); self::checkError(); return $res; }
    public function bincount(): self { $res = self::wrap(self::ffi()->tensor_bincount($this->ptr)); self::checkError(); return $res; }

    // --- CONCAT & PAD ---
    public static function concat(array $tensors, int $axis): self {
        $ffi = self::ffi();
        $num = count($tensors);
        $ptrArray = $ffi->new("TensorC*[$num]");
        foreach ($tensors as $i => $t) $ptrArray[$i] = $t->ptr;
        $res = self::wrap($ffi->tensor_concat($ptrArray, $num, $axis));
        self::checkError();
        return $res;
    }

    public function pad(array $padWidth, float $constantValue = 0.0): self {
        $ffi = self::ffi();
        $cPad = $ffi->new("int[" . count($padWidth) . "]");
        foreach ($padWidth as $i => $v) $cPad[$i] = $v;
        $res = self::wrap($ffi->tensor_pad($this->ptr, $ffi->cast("int*", $cPad), $constantValue));
        self::checkError(); return $res;
    }

    // --- SORTING & AGGREGATIONS ---
    public function argsort(int $axis = -1): self { $res = self::wrap(self::ffi()->tensor_argsort($this->ptr, $axis === -1 ? $this->ptr->ndim - 1 : $axis)); self::checkError(); return $res; }
    public function sort(int $axis = -1): self { $res = self::wrap(self::ffi()->tensor_sort($this->ptr, $axis === -1 ? $this->ptr->ndim - 1 : $axis)); self::checkError(); return $res; }
    public function topk(int $k, int $axis = -1): self { $res = self::wrap(self::ffi()->tensor_topk($this->ptr, $k, $axis === -1 ? $this->ptr->ndim - 1 : $axis)); self::checkError(); return $res; }

    public function sum(): float { $res = self::ffi()->tensor_sum($this->ptr); self::checkError(); return $res; }
    /** Sum of squared elements via BLAS sdot — no intermediate tensor allocated. */
    public function sumSquares(): float { $r = self::ffi()->tensor_sum_squares($this->ptr); self::checkError(); return (float) $r; }
    /** Temperature + top-k multinomial sample from a 1-D logit vector (all in C). */
    public static function sampleTopK(self $logits, int $k, float $temperature, int $seed = 0): int {
        $r = self::ffi()->tensor_sample_topk($logits->ptr, $k, $temperature, (int) $seed);
        self::checkError();
        return (int) $r;
    }
    public function product(): float { $res = self::ffi()->tensor_product($this->ptr); self::checkError(); return $res; }
    public function mean(): float { $res = self::ffi()->tensor_mean($this->ptr); self::checkError(); return $res; }
    public function min(): float { $res = self::ffi()->tensor_min($this->ptr); self::checkError(); return $res; }
    public function max(): float { $res = self::ffi()->tensor_max($this->ptr); self::checkError(); return $res; }
    public function argmin(): int { $res = self::ffi()->tensor_argmin($this->ptr); self::checkError(); return $res; }
    public function argmax(): int { $res = self::ffi()->tensor_argmax($this->ptr); self::checkError(); return $res; }
    public function variance(): float { $res = self::ffi()->tensor_variance($this->ptr); self::checkError(); return $res; }
    public function std(): float { $res = self::ffi()->tensor_std($this->ptr); self::checkError(); return $res; }
    public function median(): float { $res = self::ffi()->tensor_median($this->ptr); self::checkError(); return $res; }

    // --- AXIS SPECIFIC AGGREGATIONS ---
    public function sumAxis(int $axis): self { $res = self::wrap(self::ffi()->tensor_sum_axis($this->ptr, $axis)); self::checkError(); return $res; }
    public function meanAxis(int $axis): self { $res = self::wrap(self::ffi()->tensor_mean_axis($this->ptr, $axis)); self::checkError(); return $res; }
    public function maxAxis(int $axis): self { $res = self::wrap(self::ffi()->tensor_max_axis($this->ptr, $axis)); self::checkError(); return $res; }
    public function minAxis(int $axis): self { $res = self::wrap(self::ffi()->tensor_min_axis($this->ptr, $axis)); self::checkError(); return $res; }
    public function cumsum(int $axis): self { $res = self::wrap(self::ffi()->tensor_cumsum_axis($this->ptr, $axis)); self::checkError(); return $res; }

    public function sumMulti(array $axes): self {
        $ffi = self::ffi();
        $cAxes = $ffi->new("int[" . count($axes) . "]"); foreach ($axes as $i => $ax) $cAxes[$i] = $ax;
        $res = self::wrap($ffi->tensor_sum_multi($this->ptr, $ffi->cast("int*", $cAxes), count($axes)));
        self::checkError(); return $res;
    }
    public function meanMulti(array $axes): self {
        $ffi = self::ffi();
        $cAxes = $ffi->new("int[" . count($axes) . "]"); foreach ($axes as $i => $ax) $cAxes[$i] = $ax;
        $res = self::wrap($ffi->tensor_mean_multi($this->ptr, $ffi->cast("int*", $cAxes), count($axes)));
        self::checkError(); return $res;
    }
    public function maxMulti(array $axes): self {
        $ffi = self::ffi();
        $cAxes = $ffi->new("int[" . count($axes) . "]"); foreach ($axes as $i => $ax) $cAxes[$i] = $ax;
        $res = self::wrap($ffi->tensor_max_multi($this->ptr, $ffi->cast("int*", $cAxes), count($axes)));
        self::checkError(); return $res;
    }

    // --- NORMALIZATION ---
    public function normalize(): self { $res = self::wrap(self::ffi()->tensor_normalize($this->ptr)); self::checkError(); return $res; }
    public function standardize(): self { $res = self::wrap(self::ffi()->tensor_standardize($this->ptr)); self::checkError(); return $res; }
    public function normalizeInplace(): self { self::ffi()->tensor_normalize_inplace($this->ptr); self::checkError(); return $this; }
    public function standardizeInplace(): self { self::ffi()->tensor_standardize_inplace($this->ptr); self::checkError(); return $this; }

    // --- LINEAR ALGEBRA & DECOMPOSITIONS ---
    public function dot(Tensor $b): float { $res = self::ffi()->tensor_dot($this->ptr, $b->ptr); self::checkError(); return $res; }
    public function trace(): float { $res = self::ffi()->tensor_trace($this->ptr); self::checkError(); return $res; }
    /**
     * Matrix multiply. When transA or transB are true the operand is treated as
     * transposed without allocating a new tensor (BLAS handles it in one call).
     *
     * matmul($B)               → this @ B          (standard)
     * matmul($B, true)         → this^T @ B
     * matmul($B, false, true)  → this @ B^T
     */
    public function matmul(Tensor $b, bool $transA = false, bool $transB = false): self {
        $ffi = self::ffi();
        if (!$transA && !$transB) {
            $res = self::wrap($ffi->tensor_matmul($this->ptr, $b->ptr));
        } else {
            $res = self::wrap($ffi->tensor_matmul_ex($this->ptr, $b->ptr, $transA, $transB));
        }
        self::checkError(); return $res;
    }

    /**
     * Zero-allocation GEMM: $this = op(A) @ op(B)
     *
     * Writes directly into the pre-allocated receiver — no heap allocation.
     * $this must be a contiguous [m, n] tensor matching the result shape.
     *
     * Usage: $dW->matmulInto($dY, $X, transA: true)  // dW = dY^T @ X
     */
    public function matmulInto(Tensor $A, Tensor $B, bool $transA = false, bool $transB = false): void {
        self::ffi()->tensor_matmul_into($this->ptr, $A->ptr, $B->ptr, $transA, $transB);
        self::checkError();
    }

    /**
     * Zero-allocation axis-sum: $this = sum(A, axis)
     *
     * Writes into the pre-allocated receiver — no heap allocation.
     * $this must have the correct reduced shape.
     *
     * Usage: $dbias->sumAxisInto($dY, 0)  // dbias = sum(dY, axis=0)
     */
    public function sumAxisInto(Tensor $A, int $axis): void {
        self::ffi()->tensor_sum_axis_into($this->ptr, $A->ptr, $axis);
        self::checkError();
    }

    public function bmm(Tensor $b): self { $res = self::wrap(self::ffi()->tensor_bmm($this->ptr, $b->ptr)); self::checkError(); return $res; }

    // -------------------------------------------------------------------------
    // Transformer inference primitives
    // -------------------------------------------------------------------------

    /**
     * In-place row-wise RMS Layer Normalization.
     * $this must be contiguous FLOAT32; last axis is the feature dim.
     */
    public function rmsnorm(float $eps = 1e-5): void {
        self::ffi()->tensor_rmsnorm($this->ptr, $eps);
        self::checkError();
    }

    /**
     * In-place Rotary Position Embedding applied to $this (q) and $k.
     * Both must be contiguous FLOAT32; $headDim must be even.
     */
    /**
     * @param float $baseFreq Rotary base frequency (10000.0 for LLaMA-2, 500000.0 for LLaMA-3).
     * @param float $scale    Position scaling factor (1.0 = standard; <1.0 for extended context).
     */
    public function applyRope(self $k, int $headDim, int $pos,
                               float $baseFreq = 10000.0, float $scale = 1.0): void {
        self::ffi()->tensor_apply_rope($this->ptr, $k->ptr, $headDim, $pos, $baseFreq, $scale);
        self::checkError();
    }

    /**
     * In-place numerically-stable softmax along the last axis.
     * $this must be contiguous FLOAT32.
     */
    public function softmaxInplace(): void {
        self::ffi()->tensor_softmax_inplace($this->ptr);
        self::checkError();
    }

    /**
     * Scaled dot-product attention (inference, no mask).
     * $this = q, shape [seq_len, head_dim].
     * Returns a new Tensor of shape [seq_len, head_dim].
     */
    public function attention(self $k, self $v): self {
        $out = new self([$this->ptr->shape[0], $this->ptr->shape[1]]);
        self::ffi()->tensor_attention($out->ptr, $this->ptr, $k->ptr, $v->ptr);
        self::checkError();
        return $out;
    }

    /**
     * Streaming attention against a KV cache (Milakov online softmax).
     * $this = q, shape [seq_q, head_dim].
     * Returns a new Tensor of shape [seq_q, head_dim].
     */
    public function attentionKV(KVCache $cache): self {
        $seqQ = (int) ($this->ptr->total_size / $cache->ptr->head_dim);
        $out  = new self([$seqQ, $cache->ptr->head_dim]);
        self::ffi()->tensor_attention_kv($out->ptr, $this->ptr, $cache->ptr);
        self::checkError();
        return $out;
    }

    // -------------------------------------------------------------------------
    // Advanced inference & training primitives (Section 20)
    // -------------------------------------------------------------------------

    /**
     * Zero-copy tensor backed by a memory-mapped file region.
     * Use mmapFree() instead of letting the object destruct normally when
     * you want to explicitly release the mapping.
     *
     * @param string $filepath   Path to the file.
     * @param int    $byteOffset Byte offset within the file.
     * @param int[]  $shape      Tensor shape array.
     * @param int    $dtype      0=FLOAT32, 1=INT32, 2=INT64.
     */
    public static function fromMmap(string $filepath, int $byteOffset,
                                    array $shape, int $dtype = 0): self {
        $ffi  = self::ffi();
        $ndim = count($shape);
        $shapeArr = $ffi->new("int[$ndim]");
        foreach ($shape as $i => $s) $shapeArr[$i] = $s;
        $ptr = $ffi->tensor_from_mmap($filepath, $byteOffset, $ndim, $shapeArr, $dtype);
        self::checkError();
        if ($ptr === null) throw new \RuntimeException('tensor_from_mmap returned NULL.');
        $t = self::wrap($ptr);
        $t->owned = false;   /* do not call tensor_free on an mmap-backed tensor */
        return $t;
    }

    /**
     * Explicitly unmap a tensor created by fromMmap().
     * After this call the tensor's ptr is null and must not be used.
     */
    public function mmapFree(): void {
        if ($this->ptr !== null) {
            self::ffi()->tensor_mmap_free($this->ptr);
            $this->ptr  = null;
            $this->owned = false;
        }
    }

    /** SiLU activation: out[i] = x[i] * sigmoid(x[i]). AVX2 vectorized. */
    public function silu(): self {
        $res = self::wrap(self::ffi()->tensor_silu($this->ptr));
        self::checkError();
        return $res;
    }

    /**
     * Fused SwiGLU: out[i] = silu(gate[i]) * up[i].
     * $this is the gate tensor; $up must have the same shape.
     */
    public function swiglu(self $up): self {
        $res = self::wrap(self::ffi()->tensor_swiglu($this->ptr, $up->ptr));
        self::checkError();
        return $res;
    }

    /**
     * Fused cross-entropy: numerically stable softmax + NLL loss + gradient.
     * $this = logits [batch, vocab].
     * Returns ['loss' => float, 'grads' => Tensor[batch, vocab]].
     */
    public function fusedCrossEntropyLossAndGrad(self $targetIds): array {
        $ffi   = self::ffi();
        $ndim  = $this->ptr->ndim;
        $vocab = $this->ptr->shape[$ndim - 1];
        $batch = (int) ($this->ptr->total_size / $vocab);

        $shape    = [];
        for ($i = 0; $i < $ndim; $i++) $shape[] = $this->ptr->shape[$i];
        $grads    = new self($shape);
        $lossPtr  = $ffi->new('float');

        $ffi->tensor_fused_cross_entropy_loss_and_grad(
            $this->ptr, $targetIds->ptr, $grads->ptr, \FFI::addr($lossPtr)
        );
        self::checkError();

        return ['loss' => $lossPtr->cdata, 'grads' => $grads];
    }

    /**
     * RMSNorm backward.
     * $this = dY (upstream gradient) [batch, d].
     * Returns dX [batch, d].
     */
    public function rmsnormBackward(self $x, self $weights, float $eps = 1e-5): self {
        $res = self::wrap(
            self::ffi()->tensor_rmsnorm_backward($this->ptr, $x->ptr, $weights->ptr, $eps)
        );
        self::checkError();
        return $res;
    }

    /**
     * Embedding backward: accumulates $this (dY) into $dWeights for each token.
     * $dWeights must be pre-zeroed by the caller.
     * $this = dY [seq_len, embed_dim].
     */
    public function embeddingBackward(self $tokenIds, self $dWeights): void {
        self::ffi()->tensor_embedding_backward($this->ptr, $tokenIds->ptr, $dWeights->ptr);
        self::checkError();
    }
    
    public function inverse(): self { $res = self::wrap(self::ffi()->tensor_inverse($this->ptr)); self::checkError(); return $res; }
    public function pinv(): self { $res = self::wrap(self::ffi()->tensor_pinv($this->ptr)); self::checkError(); return $res; }
    public function solve(Tensor $B): self { $res = self::wrap(self::ffi()->tensor_solve($this->ptr, $B->ptr)); self::checkError(); return $res; }
    
    public function cholesky(): self { $res = self::wrap(self::ffi()->tensor_cholesky($this->ptr)); self::checkError(); return $res; }

    public function lu(): array {
        $ffi = self::ffi();
        $P = $ffi->new("TensorC*"); $L = $ffi->new("TensorC*"); $U = $ffi->new("TensorC*");
        $ffi->tensor_lu($this->ptr, \FFI::addr($P), \FFI::addr($L), \FFI::addr($U));
        self::checkError();
        return ['P' => self::wrap($P), 'L' => self::wrap($L), 'U' => self::wrap($U)];
    }

    public function svd(): array {
        $ffi = self::ffi();
        $U = $ffi->new("TensorC*"); $S = $ffi->new("TensorC*"); $Vt = $ffi->new("TensorC*");
        $ffi->tensor_svd($this->ptr, \FFI::addr($U), \FFI::addr($S), \FFI::addr($Vt));
        self::checkError();
        return ['U' => self::wrap($U), 'S' => self::wrap($S), 'Vt' => self::wrap($Vt)];
    }

    /** Economy (thin) SVD — U=[m×min_mn], Vt=[min_mn×n]. Safe on tall matrices. */
    public function svdEconomy(): array {
        $ffi = self::ffi();
        $U = $ffi->new("TensorC*"); $S = $ffi->new("TensorC*"); $Vt = $ffi->new("TensorC*");
        $ffi->tensor_svd_economy($this->ptr, \FFI::addr($U), \FFI::addr($S), \FFI::addr($Vt));
        self::checkError();
        return ['U' => self::wrap($U), 'S' => self::wrap($S), 'Vt' => self::wrap($Vt)];
    }

    public function eigenSym(): array {
        $ffi = self::ffi();
        $Vals = $ffi->new("TensorC*"); $Vecs = $ffi->new("TensorC*");
        $ffi->tensor_eigen_sym($this->ptr, \FFI::addr($Vals), \FFI::addr($Vecs));
        self::checkError();
        return ['values' => self::wrap($Vals), 'vectors' => self::wrap($Vecs)];
    }
    
    public function ref(): self { $res = self::wrap(self::ffi()->tensor_ref($this->ptr)); self::checkError(); return $res; }
    public function rref(): self { $res = self::wrap(self::ffi()->tensor_rref($this->ptr)); self::checkError(); return $res; }

    // --- DEEP LEARNING (CNN Primitives) ---
    public function im2col(int $kh, int $kw, int $sh = 1, int $sw = 1, int $ph = 0, int $pw = 0): self {
        $res = self::wrap(self::ffi()->tensor_im2col($this->ptr, $kh, $kw, $sh, $sw, $ph, $pw));
        self::checkError(); return $res;
    }
    public function col2im(int $b, int $c, int $h, int $w, int $kh, int $kw, int $sh = 1, int $sw = 1, int $ph = 0, int $pw = 0): self {
        $res = self::wrap(self::ffi()->tensor_col2im($this->ptr, $b, $c, $h, $w, $kh, $kw, $sh, $sw, $ph, $pw));
        self::checkError(); return $res;
    }
    public function conv2d(Tensor $W, ?Tensor $bias = null, int $sh = 1, int $sw = 1, int $ph = 0, int $pw = 0): self {
        $b_ptr = $bias?->ptr;
        $res = self::wrap(self::ffi()->tensor_conv2d($this->ptr, $W->ptr, $b_ptr, $sh, $sw, $ph, $pw));
        self::checkError(); return $res;
    }
    public function conv2dBackward(Tensor $X, Tensor $W, int $sh = 1, int $sw = 1, int $ph = 0, int $pw = 0): array {
        $ffi = self::ffi();
        $grads = $ffi->tensor_conv2d_backward($this->ptr, $X->ptr, $W->ptr, $sh, $sw, $ph, $pw);
        self::checkError();
        
        $dX = self::wrap($grads[0]);
        $dW = self::wrap($grads[1]);
        $dbias = self::wrap($grads[2]); 
        
        $ffi->free($grads); 
        return ['dX' => $dX, 'dW' => $dW, 'dbias' => $dbias];
    }

    // --- LLM AND NLP INTEGRATION ---
    public function embeddingLookup(Tensor $weights): self {
        $res = self::wrap(self::ffi()->tensor_embedding_lookup($this->ptr, $weights->ptr));
        self::checkError(); return $res;
    }

    // -------------------------------------------------------------------------
    // GPT TRAINING PRIMITIVES
    // -------------------------------------------------------------------------

    /** GELU activation (tanh approx). Returns new tensor with same shape. */
    public function gelu(): self {
        $res = self::wrap(self::ffi()->tensor_gelu($this->ptr));
        self::checkError(); return $res;
    }

    /** GELU backward: dx = $this * GELU'($x). $this=dOut, $x=original forward input. */
    public function geluBackward(self $x): self {
        $res = self::wrap(self::ffi()->tensor_gelu_backward($this->ptr, $x->ptr));
        self::checkError(); return $res;
    }

    /**
     * LayerNorm forward: out = (x − μ)/√(σ²+eps) · weight + bias.
     * $this: [*, D],  $weight: [D],  $bias: [D]|null,  $eps: float
     */
    public function layernormForward(self $weight, ?self $bias, float $eps = 1e-5): self {
        $res = self::wrap(self::ffi()->tensor_layernorm_forward(
            $this->ptr, $weight->ptr, $bias?->ptr, $eps
        ));
        self::checkError(); return $res;
    }

    /**
     * LayerNorm backward. $dWeight and $dBias must be pre-zeroed; accumulates +=.
     * $this=dY, returns dx.
     */
    public function layernormBackward(self $x, self $weight, float $eps,
                                       self $dWeight, ?self $dBias): self {
        $res = self::wrap(self::ffi()->tensor_layernorm_backward(
            $this->ptr, $x->ptr, $weight->ptr, $eps, $dWeight->ptr, $dBias?->ptr
        ));
        self::checkError(); return $res;
    }

    /**
     * Causal masked multi-head attention forward.
     * $this=out [nH,T,hd] pre-allocated. q,k,v: [nH,T,hd]. attn: [nH,T,T]|null.
     */
    public function causalAttention(self $q, self $k, self $v, ?self $attn): void {
        self::ffi()->tensor_causal_attention(
            $this->ptr, $q->ptr, $k->ptr, $v->ptr, $attn?->ptr
        );
        self::checkError();
    }

    /**
     * Causal attention backward.
     * $this=dOut [nH,T,hd]. dQ, dK, dV are overwritten (pre-allocated by caller).
     */
    public function causalAttentionBackward(self $attn, self $Q, self $K, self $V,
                                             self $dQ,   self $dK, self $dV): void {
        self::ffi()->tensor_causal_attention_backward(
            $this->ptr, $attn->ptr, $Q->ptr, $K->ptr, $V->ptr,
            $dQ->ptr, $dK->ptr, $dV->ptr
        );
        self::checkError();
    }

    // -------------------------------------------------------------------------
    // MAMBA / SELECTIVE SSM ENGINE
    // -------------------------------------------------------------------------

    public static function mambaAllocState(int $batch, int $dModel, int $dState): self {
        $ffi = self::ffi();
        $res = self::wrap($ffi->tensor_mamba_alloc_state($batch, $dModel, $dState));
        self::checkError();
        return $res;
    }

    public static function mambaAllocCache(int $batch, int $seqLen, int $dModel, int $dState): self {
        $ffi = self::ffi();
        $res = self::wrap($ffi->tensor_mamba_alloc_cache($batch, $seqLen, $dModel, $dState));
        self::checkError();
        return $res;
    }

    /**
     * Mamba Fused Forward
     */
    public function mambaForward(
            self $ALog, self $BProj, self $CProj, ?self $DSkip, self $delta,
            self $state, self $out, ?self $cache = null, bool $training = false
    ): void {
        self::ffi()->tensor_mamba_forward(
                $this->ptr, $ALog->ptr, $BProj->ptr, $CProj->ptr,
                $DSkip ? $DSkip->ptr : null, $delta->ptr,
                $state->ptr, $out->ptr,
                $cache ? $cache->ptr : null,
                (int) $training
        );
        self::checkError();
    }

    /**
     * Mamba Fused Backward
     */
    public function mambaBackward(
            self $x, self $ALog, self $BProj, self $CProj, ?self $DSkip, self $delta,
            self $h0, ?self $cache,
            self $dx, self $dA, self $dB, self $dC, ?self $dD, self $ddelta
    ): void {
        self::ffi()->tensor_mamba_backward(
                $this->ptr, $x->ptr, $ALog->ptr, $BProj->ptr, $CProj->ptr,
                $DSkip ? $DSkip->ptr : null, $delta->ptr,
                $h0->ptr, $cache ? $cache->ptr : null,
                $dx->ptr, $dA->ptr, $dB->ptr, $dC->ptr,
                $dD ? $dD->ptr : null, $ddelta->ptr
        );
        self::checkError();
    }

    // --- I/O SERIALIZATION ---
    public function save(string $filepath): void {
        self::ffi()->tensor_save_to_file($this->ptr, $filepath);
        self::checkError();
    }
    public static function load(string $filepath): self {
        $res = self::wrap(self::ffi()->tensor_load_from_file($filepath));
        self::checkError(); return $res;
    }

    /**
     * Save tensors in the SafeTensors format.
     *
     * @param string   $filepath    Output file path.
     * @param string   $jsonHeader  SafeTensors JSON metadata header (already serialised).
     * @param self[]   $tensors     Ordered list of Tensor objects matching the header.
     * @return int  1 on success, 0 on failure (check error via C engine).
     */
    public static function saveSafetensors(string $filepath, string $jsonHeader, array $tensors): int {
        $ffi = self::ffi();
        $num = count($tensors);
        $ptrArray = $ffi->new("TensorC*[$num]");
        foreach ($tensors as $i => $t) $ptrArray[$i] = $t->ptr;
        $jsonLen = strlen($jsonHeader);
        $res = $ffi->tensor_save_safetensors($filepath, $jsonHeader, $jsonLen, $ptrArray, $num);
        self::checkError();
        return $res;
    }

    /**
     * Load a CSV file directly into [samples, labels] Tensor pair.
     *
     * Uses the C-level two-pass CSV parser (mmap + MADV_SEQUENTIAL).
     *
     * @param string $filepath   Path to CSV file.
     * @param int    $labelCol   Column index of the label (0-based). Use -1 for no label.
     * @param bool   $hasHeader  Whether the first row is a header.
     * @return array{samples: self, labels: self|null}
     */
    public static function datasetFromCsv(string $filepath, int $labelCol = -1, bool $hasHeader = true): array {
        $ffi = self::ffi();
        $result = $ffi->tensor_dataset_from_csv($filepath, $labelCol, (int)$hasHeader);
        self::checkError();
        $samples = self::wrap($result[0]);
        $labels  = ($labelCol >= 0 && $result[1] !== null) ? self::wrap($result[1]) : null;
        $ffi->free($result);
        return ['samples' => $samples, 'labels' => $labels];
    }

    // --- DATA INGESTION ---

    /**
     * Converts a nested PHP array into a contiguous C Tensor.
     *
     * Zero-copy path: flatten in PHP → pack() to binary string → single FFI::memcpy.
     * This replaces N per-element FFI boundary crossings with exactly 1, giving
     * 10x–40x speedup on data ingestion benchmarks.
     *
     * @param array $data Deeply nested array of numbers
     * @param int $dtype defaults to Tensor::DTYPE_FLOAT32
     */
    public static function fromArray(array $data, int $dtype = self::DTYPE_FLOAT32): self {
        if (empty($data)) {
            throw new \InvalidArgumentException("Cannot create Tensor from empty array.");
        }

        // Infer shape by walking the first branch of the nested array.
        $shape = [];
        $current = $data;
        while (\is_array($current)) {
            $shape[] = \count($current);
            if (empty($current)) break;
            $current = $current[\array_key_first($current)];
        }

        $tensor = new self($shape, $dtype);
        $expectedSize = $tensor->size();

        // --- Phase 1: flatten nested array (pure PHP, no FFI) ---
        $flat = [];
        \array_walk_recursive($data, static function ($v) use (&$flat): void {
            $flat[] = $v;
        });

        $actualSize = \count($flat);
        if ($actualSize !== $expectedSize) {
            throw new \InvalidArgumentException(
                "Jagged array detected: inferred shape requires {$expectedSize} elements but found {$actualSize}."
            );
        }

        // --- Phase 2: pack binary + single FFI::memcpy (1 FFI boundary crossing) ---
        $binary = match ($dtype) {
            self::DTYPE_INT32 => \pack('l*', ...$flat),
            self::DTYPE_INT64 => \pack('q*', ...$flat),
            default           => \pack('f*', ...$flat),
        };

        \FFI::memcpy($tensor->ptr->data, $binary, \strlen($binary));

        return $tensor;
    }

    /**
     * Zero-copy read of C tensor data into a PHP array.
     *
     * Implementation notes:
     * - FFI::string() performs a single memcpy from C memory into a PHP string.
     *   No intermediate PHP array is built during the copy — it is the minimum
     *   possible data transfer for this operation.
     * - unpack() parses the binary string directly into a PHP array without
     *   any additional C↔PHP boundary crossing.
     * - For non-contiguous tensors (views, transposes) we must compact first via
     *   tensor_copy() — the copy happens in C and the result is contiguous.
     */
    public function toFlatArray(): array {
        // If non-contiguous, compact in C (single memcpy pass), then read.
        $target = $this->isContiguous() ? $this : $this->copy();

        $cdata    = $target->ptr->data;
        $byteSize = (int) $target->ptr->byte_size;
        $dtype    = $target->dtype();

        // Single FFI boundary crossing: C memory → PHP binary string.
        $binary = \FFI::string($cdata, $byteSize);

        $unpacked = match ($dtype) {
            self::DTYPE_INT32 => \unpack('l*', $binary),
            self::DTYPE_INT64 => \unpack('q*', $binary),
            default           => \unpack('f*', $binary),
        };

        // Free the compacted copy if we created one.
        if ($target !== $this) {
            unset($target);
        }

        return \array_values($unpacked);
    }
    
    public function buffer(): \FFI\CData
    {
        return $this->buffer;
    }

    /**
     * Hardware-accelerated memory copy.
     * Safely copies data from a source Tensor into this Tensor, regardless of 
     * stride differences or transpositions.
     */
    public function copyFrom(Tensor $src): void
    {
        if ($this->size() !== $src->size()) {
            throw new \InvalidArgumentException(
                "Size mismatch in copyFrom. Dest: {$this->size()}, Src: {$src->size()}"
            );
        }
        
        $ffi = self::ffi();
        // CRITICAL: Pass ->ptr (Tensor* struct), not ->buffer (float* array)
        $ffi->tensor_copy_from($this->ptr, $src->ptr);
    }
    
    // ── Section 22: Classical ML Extensions ──────────────────────────────────

    public function argmaxAxis(int $axis): self {
        $res = self::wrap(self::ffi()->tensor_argmax_axis($this->ptr, $axis));
        self::checkError(); return $res;
    }

    public static function pairwiseSqL2(self $A, self $B): self {
        $res = self::wrap(self::ffi()->tensor_pairwise_sq_l2($A->ptr, $B->ptr));
        self::checkError(); return $res;
    }

    public function expInplace(): self {
        self::ffi()->tensor_exp_inplace($this->ptr);
        self::checkError(); return $this;
    }

    public function logInplace(): self {
        self::ffi()->tensor_log_inplace($this->ptr);
        self::checkError(); return $this;
    }

    public function sqrtInplace(): self {
        self::ffi()->tensor_sqrt_inplace($this->ptr);
        self::checkError(); return $this;
    }

    public function sigmoidInplace(): self {
        self::ffi()->tensor_sigmoid_inplace($this->ptr);
        self::checkError(); return $this;
    }

    public function tanhInplace(): self {
        self::ffi()->tensor_tanh_inplace($this->ptr);
        self::checkError(); return $this;
    }

    public function reluInplace(): self {
        self::ffi()->tensor_relu_inplace($this->ptr);
        self::checkError(); return $this;
    }

    public function rowSoftmaxInplace(): self {
        self::ffi()->tensor_row_softmax_inplace($this->ptr);
        self::checkError(); return $this;
    }

    public static function gbdtComputeBoundaries(self $X, int $Q): self {
        $res = self::wrap(self::ffi()->tensor_gbdt_compute_boundaries($X->ptr, $Q));
        self::checkError(); return $res;
    }

    public static function gbdtBinSamples(self $X, self $boundaries, int $Q): self {
        $res = self::wrap(self::ffi()->tensor_gbdt_bin_samples($X->ptr, $boundaries->ptr, $Q));
        self::checkError(); return $res;
    }

    /** @return array{Tensor, Tensor} [gradients, hessians] */
    public static function gbdtMseGradHess(self $preds, self $y): array {
        $n = (int)$preds->size();
        $g = new self([$n]); $h = new self([$n]);
        self::ffi()->tensor_gbdt_mse_grad_hess($preds->ptr, $y->ptr, $g->ptr, $h->ptr);
        self::checkError();
        return [$g, $h];
    }

    /** @return array{Tensor, Tensor} [gradients, hessians] */
    public static function gbdtLogLossGradHess(self $preds, self $y): array {
        $n = (int)$preds->size();
        $g = new self([$n]); $h = new self([$n]);
        self::ffi()->tensor_gbdt_logloss_grad_hess($preds->ptr, $y->ptr, $g->ptr, $h->ptr);
        self::checkError();
        return [$g, $h];
    }

    public static function gbdtHistogram(self $bins, self $g, self $h, self $mask, int $Q): array {
        $D = $bins->shape()[1];
        $histG = new self([$D, $Q]); $histG->fill(0.0);
        $histH = new self([$D, $Q]); $histH->fill(0.0);
        self::ffi()->tensor_gbdt_histogram(
            $bins->ptr, $g->ptr, $h->ptr, $mask->ptr, $Q, $histG->ptr, $histH->ptr
        );
        self::checkError();
        return [$histG, $histH];
    }

    /** @return array{int, int, float} [feat, bin, gain]; feat=-1 means no split */
    public static function gbdtBestSplit(
        self $histG, self $histH, int $Q,
        float $sumG, float $sumH, int $nodeN,
        float $lambda, float $gamma
    ): array {
        $ffi = self::ffi();
        $feat = $ffi->new('int'); $bin = $ffi->new('int'); $gain = $ffi->new('float');
        $ffi->tensor_gbdt_best_split(
            $histG->ptr, $histH->ptr, $Q,
            $sumG, $sumH, $nodeN, $lambda, $gamma,
            \FFI::addr($feat), \FFI::addr($bin), \FFI::addr($gain)
        );
        self::checkError();
        return [(int)$feat->cdata, (int)$bin->cdata, (float)$gain->cdata];
    }

    /** @return array{Tensor, Tensor} [left_mask, right_mask] (pre-allocated, same size as mask) */
    public static function gbdtSplitNode(self $bins, self $mask, int $feat, int $bin): array {
        $n = (int)$mask->size();
        $left = new self([$n]); $right = new self([$n]);
        self::ffi()->tensor_gbdt_split_node($bins->ptr, $mask->ptr, $feat, $bin, $left->ptr, $right->ptr);
        self::checkError();
        return [$left, $right];
    }

    public static function gbdtLeafUpdate(
        self $preds, self $mask, float $sumG, float $sumH, float $lr, float $lambda
    ): float {
        $leaf = self::ffi()->tensor_gbdt_leaf_update(
            $preds->ptr, $mask->ptr, $sumG, $sumH, $lr, $lambda
        );
        self::checkError();
        return (float)$leaf;
    }

    public static function gbdtPredictAll(
        self $bins, self $feats, self $thresholds,
        self $lefts, self $rights, self $treeSizes, float $baseScore
    ): self {
        $res = self::wrap(self::ffi()->tensor_gbdt_predict_all(
            $bins->ptr, $feats->ptr, $thresholds->ptr,
            $lefts->ptr, $rights->ptr, $treeSizes->ptr, $baseScore
        ));
        self::checkError(); return $res;
    }

    public static function gbdtHistSubtract(
        self $parentG, self $parentH,
        self $siblingG, self $siblingH,
        self $outG, self $outH
    ): void {
        self::ffi()->tensor_gbdt_hist_subtract(
            $parentG->ptr, $parentH->ptr,
            $siblingG->ptr, $siblingH->ptr,
            $outG->ptr, $outH->ptr
        );
        self::checkError();
    }

    /**
     * Leaf-wise GBDT tree training (LightGBM-style).
     * Updates $preds in-place (+= lr * leaf_value for each sample).
     * Writes tree structure into $outFeats/$outThresh/$outLefts/$outRights.
     * Returns node count for this tree.
     *
     * @param int   $Q         number of bins
     * @param int   $maxLeaves max leaf nodes (= 2^maxDepth)
     * @param float $alpha     L1 regularisation (0 = disabled)
     */
    public static function gbdtTrainTree(
        self $bins, self $g, self $h,
        int $Q, int $maxLeaves,
        float $lambda, float $alpha, float $gamma, float $minHess, float $lr,
        self $preds,
        self $outFeats, self $outThresh, self $outLefts, self $outRights
    ): int {
        $n = self::ffi()->tensor_gbdt_train_tree(
            $bins->ptr, $g->ptr, $h->ptr, $Q, $maxLeaves,
            $lambda, $alpha, $gamma, $minHess, $lr,
            $preds->ptr,
            $outFeats->ptr, $outThresh->ptr, $outLefts->ptr, $outRights->ptr
        );
        self::checkError();
        return (int)$n;
    }

    public static function quantileFit(self $X, int $nQuantiles = 1000): self {
        $res = self::wrap(self::ffi()->tensor_quantile_fit($X->ptr, $nQuantiles));
        self::checkError(); return $res;
    }

    public static function quantileTransform(self $X, self $landmarks): self {
        $nq = $landmarks->shape()[1];
        $res = self::wrap(self::ffi()->tensor_quantile_transform($X->ptr, $landmarks->ptr, $nq));
        self::checkError(); return $res;
    }

    public static function yjFit(self $X): self {
        $res = self::wrap(self::ffi()->tensor_yj_fit($X->ptr));
        self::checkError(); return $res;
    }

    public static function yjTransform(self $X, self $lambdas): self {
        $res = self::wrap(self::ffi()->tensor_yj_transform($X->ptr, $lambdas->ptr));
        self::checkError(); return $res;
    }

    /** Replace NaN/Inf values in-place with fill_val (FLOAT32 only). */
    public function fillNan(float $fillVal = 0.0): self {
        self::ffi()->tensor_fill_nan($this->ptr, $fillVal);
        self::checkError(); return $this;
    }

    /**
     * Compute Pearson r between each column of $this [N,D] and $y [N].
     * Returns [D] FLOAT32 tensor. NaN pairs are skipped (pairwise-complete).
     */
    public function pearsonCols(self $y): self {
        $res = self::wrap(self::ffi()->tensor_pearson_cols($this->ptr, $y->ptr));
        self::checkError(); return $res;
    }

    // ── Section 24: HPC Estimator Kernels ────────────────────────────────────

    /** [N] float32 class indices → [N, K] float32 one-hot matrix. */
    public static function onehot(self $indices, int $numClasses): self {
        $res = self::wrap(self::ffi()->tensor_onehot($indices->ptr, $numClasses));
        self::checkError(); return $res;
    }

    /** KNN majority vote: [N,k] neighbor labels → [N] predicted class (float32 class index). */
    public static function knnVote(self $kLabels, int $numClasses): self {
        $res = self::wrap(self::ffi()->tensor_knn_vote($kLabels->ptr, $numClasses));
        self::checkError(); return $res;
    }

    /** KMeans E-step: X[N,D] × centroids[K,D] → [N] cluster assignment indices. */
    public static function kmeansAssign(self $X, self $centroids): self {
        $res = self::wrap(self::ffi()->tensor_kmeans_assign($X->ptr, $centroids->ptr));
        self::checkError(); return $res;
    }

    /** KMeans M-step: X[N,D] × assignments[N] → [K,D] new centroids.
     *  Empty clusters retain the corresponding row of $oldCentroids (pass null to zero them). */
    public static function kmeansCentroids(self $X, self $assignments, int $K,
                                           ?self $oldCentroids = null): self {
        $res = self::wrap(self::ffi()->tensor_kmeans_centroids(
            $X->ptr, $assignments->ptr, $K, $oldCentroids?->ptr
        ));
        self::checkError(); return $res;
    }

    /** Closed-form Ridge Regression: W = (X^T X + λI)^{-1} X^T y → [D,1]. */
    public static function ridgeSolve(self $X, self $y, float $lambda = 1.0): self {
        $res = self::wrap(self::ffi()->tensor_ridge_solve($X->ptr, $y->ptr, $lambda));
        self::checkError(); return $res;
    }

    /** Copy per-tree scratch node array into flat ensemble buffer (replaces PHP buffer copy loop). */
    public static function gbdtCollectTree(self $dest, int $treeIdx, int $maxNodes, self $src): void {
        self::ffi()->tensor_gbdt_collect_tree($dest->ptr, $treeIdx, $maxNodes, $src->ptr);
        self::checkError();
    }

    // ── Section 26: Multiclass GBDT Kernels ──────────────────────────────────

    /** Broadcast [K] base scores into every row of pre-allocated [N,K] preds tensor. */
    public static function gbdtInitPredsMC(self $outNK, self $baseK): void {
        self::ffi()->tensor_gbdt_init_preds_mc($outNK->ptr, $baseK->ptr);
        self::checkError();
    }

    /**
     * Softmax cross-entropy gradients/hessians for all K classes in one C call.
     * rawNK: [N,K] FLOAT32 logits; yN: [N] INT32 or FLOAT32 class indices.
     * Returns [outG [N,K], outH [N,K]] pre-allocated and filled by C.
     *
     * @return array{Tensor, Tensor}
     */
    public static function gbdtSoftmaxGradHess(self $rawNK, self $yN): array {
        $shape = $rawNK->shape();
        $outG = new self($shape);
        $outH = new self($shape);
        self::ffi()->tensor_gbdt_softmax_grad_hess($rawNK->ptr, $yN->ptr, $outG->ptr, $outH->ptr);
        self::checkError();
        return [$outG, $outH];
    }

    /**
     * In-place variant: writes into caller-supplied $outG / $outH (same shape as $rawNK).
     * Avoids re-allocating the gradient tensors every boosting round.
     */
    public static function gbdtSoftmaxGradHessInto(
        self $rawNK, self $yN, self $outG, self $outH
    ): void {
        self::ffi()->tensor_gbdt_softmax_grad_hess($rawNK->ptr, $yN->ptr, $outG->ptr, $outH->ptr);
        self::checkError();
    }

    /**
     * Train one leaf-wise GBDT tree for class column $kc of [N,K] grad/hess tensors.
     * Reads g_NK/h_NK/preds_NK at stride K — zero memory copies.
     * Updates preds_NK in-place for column $kc only.
     * Returns number of nodes used.
     */
    public static function gbdtTrainTreeMC(
        self $bins, self $gNK, self $hNK,
        int $K, int $kc,
        int $Q, int $maxLeaves,
        float $lambda, float $alpha, float $gamma, float $minHess, float $lr,
        self $predsNK,
        self $outFeats, self $outThresh, self $outLefts, self $outRights
    ): int {
        $n = self::ffi()->tensor_gbdt_train_tree_mc(
            $bins->ptr, $gNK->ptr, $hNK->ptr, $K, $kc,
            $Q, $maxLeaves, $lambda, $alpha, $gamma, $minHess, $lr,
            $predsNK->ptr,
            $outFeats->ptr, $outThresh->ptr, $outLefts->ptr, $outRights->ptr
        );
        self::checkError();
        return (int)$n;
    }

    /**
     * Multiclass batch prediction: one FFI call returns [N, K] raw logits.
     * feats/thresh/lefts/rights: [T*K, maxNodes]; treeSizes: [T*K]; baseScores: [K].
     * Apply rowSoftmaxInplace() for class probabilities.
     */
    public static function gbdtPredictAllMC(
        self $bins, self $feats, self $thresholds,
        self $lefts, self $rights, self $treeSizes,
        self $baseScores, int $K
    ): self {
        $res = self::wrap(self::ffi()->tensor_gbdt_predict_all_mc(
            $bins->ptr, $feats->ptr, $thresholds->ptr,
            $lefts->ptr, $rights->ptr, $treeSizes->ptr,
            $baseScores->ptr, $K
        ));
        self::checkError();
        return $res;
    }

    /** Isolation Forest batch scoring: returns [N] anomaly scores in [0,1].
     *  All tree arrays must be pre-flattened to [T * maxNodes] via serializeIforestTrees(). */
    public static function iforestScore(self $X,
                                        self $featsFlat, self $threshFlat,
                                        self $leftsFlat, self $rightsFlat,
                                        self $lsizeFlat, self $treeSizes,
                                        float $cNorm): self {
        $res = self::wrap(self::ffi()->tensor_iforest_score(
            $X->ptr,
            $featsFlat->ptr, $threshFlat->ptr,
            $leftsFlat->ptr, $rightsFlat->ptr,
            $lsizeFlat->ptr, $treeSizes->ptr,
            $cNorm
        ));
        self::checkError(); return $res;
    }

    // ── Section 25: HPC Estimator Kernels — Batch 2 ─────────────────────────

    /** Returns [N] float32 bootstrap indices in [0, N-1] (with replacement). */
    public static function bootstrapIndices(int $n): self {
        $res = self::wrap(self::ffi()->tensor_bootstrap_indices($n));
        self::checkError(); return $res;
    }

    /** Majority vote over [N, T] integer-label matrix → [N]. */
    public static function matrixVote(self $votes, int $numClasses): self {
        $res = self::wrap(self::ffi()->tensor_matrix_vote($votes->ptr, $numClasses));
        self::checkError(); return $res;
    }

    /**
     * All-C CART split search.
     * Returns [N+2]: [best_feature(-1=none), best_threshold, mask[0..N-1]].
     */
    public static function cartFindSplit(self $X, self $y, self $featureIndices,
                                          int $numThresholds = 8): self {
        $res = self::wrap(self::ffi()->tensor_cart_find_split(
            $X->ptr, $y->ptr, $featureIndices->ptr, $numThresholds
        ));
        self::checkError(); return $res;
    }

    /** Fused (Elastic)Net SGD step: updates W [D,1] and bias [1] in-place.
     *  l1Ratio=1.0→Lasso, 0.0→Ridge SGD, 0.5→ElasticNet. */
    public static function lassoSgdStep(self $X, self $y, self $W, self $bias,
                                         float $alpha, float $lr,
                                         float $l1Ratio = 1.0): void {
        self::ffi()->tensor_lasso_sgd_step(
            $X->ptr, $y->ptr, $W->ptr, $bias->ptr, $alpha, $lr, $l1Ratio
        );
        self::checkError();
    }

    /**
     * Fused GNB log-likelihood: returns [N, K].
     * logNormsK[k] = log_prior[k] − 0.5·Σ_d log(2π·var[k,d])
     */
    public static function gnbLogLikelihood(self $X, self $meansKD,
                                             self $varsKD, self $logNormsK): self {
        $res = self::wrap(self::ffi()->tensor_gnb_log_likelihood(
            $X->ptr, $meansKD->ptr, $varsKD->ptr, $logNormsK->ptr
        ));
        self::checkError(); return $res;
    }

    /** Gather: out[i] = table[indices[i]]. Replaces PHP array_map remapping. */
    public static function gatherIndices(self $indices, self $table): self {
        $res = self::wrap(self::ffi()->tensor_gather_indices($indices->ptr, $table->ptr));
        self::checkError(); return $res;
    }

    public function __destruct()
    {
        if ($this->owned && $this->ptr !== null) {
            // Memory safe call: if this tensor was born in an Arena,
            // tensor_free safely intercepts it and does nothing!
            $ffi = self::ffi();
            $ffi->tensor_free($this->ptr); // @phpstan-ignore-line — FFI methods are resolved at runtime
            $this->ptr = null;
            $this->buffer = null;
        }
    }
}