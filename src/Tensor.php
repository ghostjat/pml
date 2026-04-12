<?php
declare(strict_types=1);

namespace Pml;
use Pml\Lib\TensorEngine;

/**
 * The PHP PyTorch/NumPy equivalent API.
 */
class Tensor {
    // Data Type Enums mapping to the C definitions
    public const DTYPE_FLOAT32 = 0;
    public const DTYPE_INT32 = 1;
    public const DTYPE_INT64 = 2;

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

        $ffi = TensorEngine::get();
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

        if ($dtype === self::DTYPE_INT32) {
            $this->buffer = $ffi->cast("int32_t*", $this->ptr->data);
        } elseif ($dtype === self::DTYPE_INT64) {
            $this->buffer = $ffi->cast("int64_t*", $this->ptr->data);
        } else {
            $this->buffer = $ffi->cast("float*", $this->ptr->data);
        }
    }

    public static function wrap(?\FFI\CData $ptr, ?self $parent = null): self {
        if ($ptr === null) {
            self::checkError();
            throw new \RuntimeException("C-Engine returned NULL without setting error.");
        }

        $ref = new \ReflectionClass(self::class);
        $t = $ref->newInstanceWithoutConstructor();
        $t->ptr = $ptr;
        
        $ffi = TensorEngine::get();
        $dtype = $ptr->dtype;
        
        if ($dtype === self::DTYPE_INT32) {
            $t->buffer = $ffi->cast("int32_t*", $ptr->data);
        } elseif ($dtype === self::DTYPE_INT64) {
            $t->buffer = $ffi->cast("int64_t*", $ptr->data);
        } else {
            $t->buffer = $ffi->cast("float*", $ptr->data);
        }
        
        $t->owned = true; 
        $t->parent = $parent; 
        return $t;
    }

    /**
     * Translates a C-engine error into a PHP RuntimeException gracefully.
     */
    private static function checkError(): void {
        $ffi = TensorEngine::get();
        if ($ffi->tensor_check_error()) {
            // Clear BEFORE reading the string so the error never leaks into
            // subsequent tests even if FFI::string() itself throws.
            $ffi->tensor_clear_error();
            $errPtr = $ffi->tensor_get_last_error();
            // PHP FFI may return const char* as either FFI\CData or a plain PHP
            // string depending on the runtime version.
            $err = $errPtr instanceof \FFI\CData ? \FFI::string($errPtr) : (string)$errPtr;
            throw new \RuntimeException("C-Engine Error: {$err}");
        }
    }

    // --- METADATA GETTERS ---
    public function dtype(): int { return $this->ptr->dtype; }
    public function shape(): array {
        $shape = [];
        for ($i = 0; $i < $this->ptr->ndim; $i++) $shape[] = $this->ptr->shape[$i];
        return $shape;
    }
    public function size(): int { return $this->ptr->total_size; }
    public function ndim(): int { return $this->ptr->ndim; }

    // --- ZERO-COPY VIEWS & SLICING ---
    public function view(): self { 
        $res = self::wrap(TensorEngine::get()->tensor_view($this->ptr), $this); 
        self::checkError(); return $res;
    }
    public function slice(int $axis, int $start, int $length): self { 
        $res = self::wrap(TensorEngine::get()->tensor_slice($this->ptr, $axis, $start, $length), $this); 
        self::checkError(); return $res;
    }
    public function sliceStep(int $axis, int $start, int $end, int $step): self { 
        $res = self::wrap(TensorEngine::get()->tensor_slice_step($this->ptr, $axis, $start, $end, $step), $this); 
        self::checkError(); return $res;
    }
    public function row(int $row): self { 
        $res = self::wrap(TensorEngine::get()->tensor_row_view($this->ptr, $row), $this); 
        self::checkError(); return $res;
    }
    public function col(int $col): self { 
        $res = self::wrap(TensorEngine::get()->tensor_column_view($this->ptr, $col), $this); 
        self::checkError(); return $res;
    }

    // --- INITIALIZERS & CREATION ---
    public function fill(float $val): self { 
        TensorEngine::get()->tensor_fill($this->ptr, $val); 
        self::checkError(); return $this; 
    }
    public function copy(): self { 
        $res = self::wrap(TensorEngine::get()->tensor_copy($this->ptr)); 
        self::checkError(); return $res;
    }
    public function isContiguous(): bool { 
        $res = TensorEngine::get()->tensor_is_contiguous($this->ptr); 
        self::checkError(); return $res;
    }
    
    public static function zeros(int ...$shape): self {
        $ndim = count($shape); $ffi = TensorEngine::get();
        $cShape = $ffi->new("int[$ndim]"); foreach ($shape as $i => $dim) $cShape[$i] = $dim;
        $res = self::wrap($ffi->tensor_zeros($ndim, $ffi->cast("int*", $cShape)));
        self::checkError(); return $res;
    }

    public static function ones(int ...$shape): self {
        $ndim = count($shape); $ffi = TensorEngine::get();
        $cShape = $ffi->new("int[$ndim]"); foreach ($shape as $i => $dim) $cShape[$i] = $dim;
        $res = self::wrap($ffi->tensor_ones($ndim, $ffi->cast("int*", $cShape)));
        self::checkError(); return $res;
    }

    public static function zerosArena(\FFI\CData $arena, int ...$shape): self {
        $t = new self($shape, self::DTYPE_FLOAT32, $arena);
        TensorEngine::get()->tensor_fill($t->ptr, 0.0);
        self::checkError();
        return $t;
    }

    public static function onesArena(\FFI\CData $arena, int ...$shape): self {
        $t = new self($shape, self::DTYPE_FLOAT32, $arena);
        TensorEngine::get()->tensor_fill($t->ptr, 1.0);
        self::checkError();
        return $t;
    }

    public static function range(float $start, float $end, float $step = 1.0): self { 
        $res = self::wrap(TensorEngine::get()->tensor_range($start, $end, $step)); 
        self::checkError(); return $res;
    }
    public static function linspace(float $start, float $end, int $steps): self { 
        $res = self::wrap(TensorEngine::get()->tensor_linspace($start, $end, $steps)); 
        self::checkError(); return $res;
    }

    public static function randomNormal(array $shape, float $mean = 0.0, float $stddev = 1.0, ?\FFI\CData $arena = null): self {
        $t = new self($shape, self::DTYPE_FLOAT32, $arena); 
        TensorEngine::get()->tensor_random_normal($t->ptr, $mean, $stddev); 
        self::checkError(); return $t;
    }
    public static function randomUniform(array $shape, float $minVal = 0.0, float $maxVal = 1.0, ?\FFI\CData $arena = null): self {
        $t = new self($shape, self::DTYPE_FLOAT32, $arena); 
        TensorEngine::get()->tensor_random_uniform($t->ptr, $minVal, $maxVal); 
        self::checkError(); return $t;
    }
    public function randomChoice(int $n, bool $replace = true): self { 
        $res = self::wrap(TensorEngine::get()->tensor_random_choice($this->ptr, $n, $replace)); 
        self::checkError(); return $res;
    }
    public function randomPermutation(): self { 
        $res = self::wrap(TensorEngine::get()->tensor_random_permutation($this->ptr)); 
        self::checkError(); return $res;
    }

    // --- FUSED KERNELS & HARDWARE INFERENCE ---
    
    public static function fusedBceLossAndGrad(self $preds, self $targets, ?self $grads = null): float {
        $ffi = TensorEngine::get();
        $outLoss = $ffi->new("float[1]"); 
        $gradsPtr = $grads?->ptr;
        
        $ffi->tensor_fused_bce_loss_and_grad($preds->ptr, $targets->ptr, $gradsPtr, \FFI::addr($outLoss[0]));
        self::checkError();
        return $outLoss[0];
    }

    public static function fusedAdamStep(self $param, self $grad, self $m, self $v, float $lr, float $b1, float $b2, float $eps, int $t): void {
        TensorEngine::get()->tensor_fused_adam_step($param->ptr, $grad->ptr, $m->ptr, $v->ptr, $lr, $b1, $b2, $eps, $t);
        self::checkError();
    }

    // --- SHAPE MUTATIONS ---
    public function reshape(int ...$newShape): self {
        $ndim = count($newShape); $ffi = TensorEngine::get();
        $cShape = $ffi->new("int[$ndim]"); foreach ($newShape as $i => $dim) $cShape[$i] = $dim;
        $res = self::wrap($ffi->tensor_reshape($this->ptr, $ndim, $ffi->cast("int*", $cShape)), $this);
        self::checkError(); return $res;
    }
    public function flatten(): self { 
        $res = self::wrap(TensorEngine::get()->tensor_flatten($this->ptr), $this); 
        self::checkError(); return $res;
    }
    public function expandDims(int $axis): self { 
        $res = self::wrap(TensorEngine::get()->tensor_expand_dims($this->ptr, $axis), $this); 
        self::checkError(); return $res;
    }
    public function squeeze(): self { 
        $res = self::wrap(TensorEngine::get()->tensor_squeeze($this->ptr), $this); 
        self::checkError(); return $res;
    }
    public function transpose(): self { 
        $res = self::wrap(TensorEngine::get()->tensor_transpose_2d($this->ptr), $this); 
        self::checkError(); return $res;
    }
    
    public function transposeNd(array $axes): self {
        $ffi = TensorEngine::get();
        $cAxes = $ffi->new("int[" . count($axes) . "]"); foreach ($axes as $i => $ax) $cAxes[$i] = $ax;
        $res = self::wrap($ffi->tensor_transpose_nd($this->ptr, $ffi->cast("int*", $cAxes)), $this);
        self::checkError(); return $res;
    }
    public function swapaxes(int $axis1, int $axis2): self { 
        $res = self::wrap(TensorEngine::get()->tensor_swapaxes($this->ptr, $axis1, $axis2), $this); 
        self::checkError(); return $res;
    }

    // --- MATH BINARY & SCALAR ---
    public function add(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_add($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function sub(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_sub($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function mul(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_mul($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function div(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_div($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function addScalar(float $val): self { $res = self::wrap(TensorEngine::get()->tensor_add_scalar($this->ptr, $val)); self::checkError(); return $res; }
    public function mulScalar(float $val): self { $res = self::wrap(TensorEngine::get()->tensor_mul_scalar($this->ptr, $val)); self::checkError(); return $res; }
    public function pow(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_pow($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function clip(float $min, float $max): self { $res = self::wrap(TensorEngine::get()->tensor_clip($this->ptr, $min, $max)); self::checkError(); return $res; }

    // O(1) FFI Logic
    public function lessScalar(float $val): self { $res = self::wrap(TensorEngine::get()->tensor_less_scalar_f32($this->ptr, $val)); self::checkError(); return $res; }
    public function greaterScalar(float $val): self { $res = self::wrap(TensorEngine::get()->tensor_greater_scalar_f32($this->ptr, $val)); self::checkError(); return $res; }

    // --- IN-PLACE OPS ---
    public function addInplace(Tensor $b): self { TensorEngine::get()->tensor_add_inplace($this->ptr, $b->ptr); self::checkError(); return $this; }
    public function subInplace(Tensor $b): self { TensorEngine::get()->tensor_sub_inplace($this->ptr, $b->ptr); self::checkError(); return $this; }
    public function mulInplace(Tensor $b): self { TensorEngine::get()->tensor_mul_inplace($this->ptr, $b->ptr); self::checkError(); return $this; }
    public function divInplace(Tensor $b): self { TensorEngine::get()->tensor_div_inplace($this->ptr, $b->ptr); self::checkError(); return $this; }
    public function addScalarInplace(float $val): self { TensorEngine::get()->tensor_add_scalar_inplace($this->ptr, $val); self::checkError(); return $this; }
    public function mulScalarInplace(float $val): self { TensorEngine::get()->tensor_mul_scalar_inplace($this->ptr, $val); self::checkError(); return $this; }

    // --- MATH UNARY ---
    public function sqrt(): self { $res = self::wrap(TensorEngine::get()->tensor_sqrt($this->ptr)); self::checkError(); return $res; }
    public function square(): self { $res = self::wrap(TensorEngine::get()->tensor_square($this->ptr)); self::checkError(); return $res; }
    public function abs(): self { $res = self::wrap(TensorEngine::get()->tensor_abs($this->ptr)); self::checkError(); return $res; }
    public function sign(): self { $res = self::wrap(TensorEngine::get()->tensor_sign($this->ptr)); self::checkError(); return $res; }
    public function exp(): self { $res = self::wrap(TensorEngine::get()->tensor_exp($this->ptr)); self::checkError(); return $res; }
    public function log(): self { $res = self::wrap(TensorEngine::get()->tensor_log($this->ptr)); self::checkError(); return $res; }
    public function log1p(): self { $res = self::wrap(TensorEngine::get()->tensor_log1p($this->ptr)); self::checkError(); return $res; }
    public function round(): self { $res = self::wrap(TensorEngine::get()->tensor_round($this->ptr)); self::checkError(); return $res; }
    public function floor(): self { $res = self::wrap(TensorEngine::get()->tensor_floor($this->ptr)); self::checkError(); return $res; }
    public function ceil(): self { $res = self::wrap(TensorEngine::get()->tensor_ceil($this->ptr)); self::checkError(); return $res; }
    public function sigmoid(): self { $res = self::wrap(TensorEngine::get()->tensor_sigmoid($this->ptr)); self::checkError(); return $res; }
    public function tanh(): self { $res = self::wrap(TensorEngine::get()->tensor_tanh($this->ptr)); self::checkError(); return $res; }
    public function relu(): self { $res = self::wrap(TensorEngine::get()->tensor_relu($this->ptr)); self::checkError(); return $res; }
    public function sin(): self { $res = self::wrap(TensorEngine::get()->tensor_sin($this->ptr)); self::checkError(); return $res; }
    public function cos(): self { $res = self::wrap(TensorEngine::get()->tensor_cos($this->ptr)); self::checkError(); return $res; }
    public function tan(): self { $res = self::wrap(TensorEngine::get()->tensor_tan($this->ptr)); self::checkError(); return $res; }
    public function asin(): self { $res = self::wrap(TensorEngine::get()->tensor_asin($this->ptr)); self::checkError(); return $res; }
    public function acos(): self { $res = self::wrap(TensorEngine::get()->tensor_acos($this->ptr)); self::checkError(); return $res; }
    public function atan(): self { $res = self::wrap(TensorEngine::get()->tensor_atan($this->ptr)); self::checkError(); return $res; }

    // --- LOGICAL & MESSY DATA ---
    public function equal(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_equal($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function notEqual(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_not_equal($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function greater(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_greater($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function greaterEqual(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_greater_equal($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function less(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_less($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function lessEqual(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_less_equal($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function logicalNot(): self { $res = self::wrap(TensorEngine::get()->tensor_logical_not($this->ptr)); self::checkError(); return $res; }
    
    public function isNan(): self { $res = self::wrap(TensorEngine::get()->tensor_isnan($this->ptr)); self::checkError(); return $res; }
    public function isInf(): self { $res = self::wrap(TensorEngine::get()->tensor_isinf($this->ptr)); self::checkError(); return $res; }
    public function nanToNumInplace(float $nan_val = 0.0, float $posinf = 1e38, float $neginf = -1e38): self { TensorEngine::get()->tensor_nan_to_num_inplace($this->ptr, $nan_val, $posinf, $neginf); self::checkError(); return $this; }
    public function any(): bool { $res = TensorEngine::get()->tensor_any($this->ptr); self::checkError(); return $res; }
    public function all(): bool { $res = TensorEngine::get()->tensor_all($this->ptr); self::checkError(); return $res; }

    // --- FANCY INDEXING & SETS ---
    public function where(Tensor $x, Tensor $y): self { $res = self::wrap(TensorEngine::get()->tensor_where($this->ptr, $x->ptr, $y->ptr)); self::checkError(); return $res; }
    public function booleanIndex(Tensor $mask): self { $res = self::wrap(TensorEngine::get()->tensor_boolean_index($this->ptr, $mask->ptr)); self::checkError(); return $res; }
    public function take(Tensor $indices, int $axis): self { $res = self::wrap(TensorEngine::get()->tensor_take($this->ptr, $indices->ptr, $axis)); self::checkError(); return $res; }
    public function unique(): self { $res = self::wrap(TensorEngine::get()->tensor_unique($this->ptr)); self::checkError(); return $res; }
    public function bincount(): self { $res = self::wrap(TensorEngine::get()->tensor_bincount($this->ptr)); self::checkError(); return $res; }

    // --- CONCAT & PAD ---
    public static function concat(array $tensors, int $axis): self {
        $ffi = TensorEngine::get();
        $num = count($tensors);
        $ptrArray = $ffi->new("TensorC*[$num]");
        foreach ($tensors as $i => $t) $ptrArray[$i] = $t->ptr;
        $res = self::wrap($ffi->tensor_concat($ptrArray, $num, $axis));
        self::checkError();
        return $res;
    }

    public function pad(array $padWidth, float $constantValue = 0.0): self {
        $ffi = TensorEngine::get();
        $cPad = $ffi->new("int[" . count($padWidth) . "]");
        foreach ($padWidth as $i => $v) $cPad[$i] = $v;
        $res = self::wrap($ffi->tensor_pad($this->ptr, $ffi->cast("int*", $cPad), $constantValue));
        self::checkError(); return $res;
    }

    // --- SORTING & AGGREGATIONS ---
    public function argsort(int $axis = -1): self { $res = self::wrap(TensorEngine::get()->tensor_argsort($this->ptr, $axis === -1 ? $this->ptr->ndim - 1 : $axis)); self::checkError(); return $res; }
    public function sort(int $axis = -1): self { $res = self::wrap(TensorEngine::get()->tensor_sort($this->ptr, $axis === -1 ? $this->ptr->ndim - 1 : $axis)); self::checkError(); return $res; }
    public function topk(int $k, int $axis = -1): self { $res = self::wrap(TensorEngine::get()->tensor_topk($this->ptr, $k, $axis === -1 ? $this->ptr->ndim - 1 : $axis)); self::checkError(); return $res; }

    public function sum(): float { $res = TensorEngine::get()->tensor_sum($this->ptr); self::checkError(); return $res; }
    public function product(): float { $res = TensorEngine::get()->tensor_product($this->ptr); self::checkError(); return $res; }
    public function mean(): float { $res = TensorEngine::get()->tensor_mean($this->ptr); self::checkError(); return $res; }
    public function min(): float { $res = TensorEngine::get()->tensor_min($this->ptr); self::checkError(); return $res; }
    public function max(): float { $res = TensorEngine::get()->tensor_max($this->ptr); self::checkError(); return $res; }
    public function argmin(): int { $res = TensorEngine::get()->tensor_argmin($this->ptr); self::checkError(); return $res; }
    public function argmax(): int { $res = TensorEngine::get()->tensor_argmax($this->ptr); self::checkError(); return $res; }
    public function variance(): float { $res = TensorEngine::get()->tensor_variance($this->ptr); self::checkError(); return $res; }
    public function std(): float { $res = TensorEngine::get()->tensor_std($this->ptr); self::checkError(); return $res; }
    public function median(): float { $res = TensorEngine::get()->tensor_median($this->ptr); self::checkError(); return $res; }

    // --- AXIS SPECIFIC AGGREGATIONS ---
    public function sumAxis(int $axis): self { $res = self::wrap(TensorEngine::get()->tensor_sum_axis($this->ptr, $axis)); self::checkError(); return $res; }
    public function meanAxis(int $axis): self { $res = self::wrap(TensorEngine::get()->tensor_mean_axis($this->ptr, $axis)); self::checkError(); return $res; }
    public function maxAxis(int $axis): self { $res = self::wrap(TensorEngine::get()->tensor_max_axis($this->ptr, $axis)); self::checkError(); return $res; }
    public function minAxis(int $axis): self { $res = self::wrap(TensorEngine::get()->tensor_min_axis($this->ptr, $axis)); self::checkError(); return $res; }
    public function cumsum(int $axis): self { $res = self::wrap(TensorEngine::get()->tensor_cumsum_axis($this->ptr, $axis)); self::checkError(); return $res; }

    public function sumMulti(array $axes): self {
        $ffi = TensorEngine::get();
        $cAxes = $ffi->new("int[" . count($axes) . "]"); foreach ($axes as $i => $ax) $cAxes[$i] = $ax;
        $res = self::wrap($ffi->tensor_sum_multi($this->ptr, $ffi->cast("int*", $cAxes), count($axes)));
        self::checkError(); return $res;
    }
    public function meanMulti(array $axes): self {
        $ffi = TensorEngine::get();
        $cAxes = $ffi->new("int[" . count($axes) . "]"); foreach ($axes as $i => $ax) $cAxes[$i] = $ax;
        $res = self::wrap($ffi->tensor_mean_multi($this->ptr, $ffi->cast("int*", $cAxes), count($axes)));
        self::checkError(); return $res;
    }

    // --- NORMALIZATION ---
    public function normalize(): self { $res = self::wrap(TensorEngine::get()->tensor_normalize($this->ptr)); self::checkError(); return $res; }
    public function standardize(): self { $res = self::wrap(TensorEngine::get()->tensor_standardize($this->ptr)); self::checkError(); return $res; }
    public function normalizeInplace(): self { TensorEngine::get()->tensor_normalize_inplace($this->ptr); self::checkError(); return $this; }
    public function standardizeInplace(): self { TensorEngine::get()->tensor_standardize_inplace($this->ptr); self::checkError(); return $this; }

    // --- LINEAR ALGEBRA & DECOMPOSITIONS ---
    public function dot(Tensor $b): float { $res = TensorEngine::get()->tensor_dot($this->ptr, $b->ptr); self::checkError(); return $res; }
    public function trace(): float { $res = TensorEngine::get()->tensor_trace($this->ptr); self::checkError(); return $res; }
    public function matmul(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_matmul($this->ptr, $b->ptr)); self::checkError(); return $res; }
    public function bmm(Tensor $b): self { $res = self::wrap(TensorEngine::get()->tensor_bmm($this->ptr, $b->ptr)); self::checkError(); return $res; }
    
    public function inverse(): self { $res = self::wrap(TensorEngine::get()->tensor_inverse($this->ptr)); self::checkError(); return $res; }
    public function pinv(): self { $res = self::wrap(TensorEngine::get()->tensor_pinv($this->ptr)); self::checkError(); return $res; }
    public function solve(Tensor $B): self { $res = self::wrap(TensorEngine::get()->tensor_solve($this->ptr, $B->ptr)); self::checkError(); return $res; }
    
    public function cholesky(): self { $res = self::wrap(TensorEngine::get()->tensor_cholesky($this->ptr)); self::checkError(); return $res; }

    public function lu(): array {
        $ffi = TensorEngine::get();
        $P = $ffi->new("TensorC*"); $L = $ffi->new("TensorC*"); $U = $ffi->new("TensorC*");
        $ffi->tensor_lu($this->ptr, \FFI::addr($P), \FFI::addr($L), \FFI::addr($U));
        self::checkError();
        return ['P' => self::wrap($P), 'L' => self::wrap($L), 'U' => self::wrap($U)];
    }

    public function svd(): array {
        $ffi = TensorEngine::get();
        $U = $ffi->new("TensorC*"); $S = $ffi->new("TensorC*"); $Vt = $ffi->new("TensorC*");
        $ffi->tensor_svd($this->ptr, \FFI::addr($U), \FFI::addr($S), \FFI::addr($Vt));
        self::checkError();
        return ['U' => self::wrap($U), 'S' => self::wrap($S), 'Vt' => self::wrap($Vt)];
    }

    public function eigenSym(): array {
        $ffi = TensorEngine::get();
        $Vals = $ffi->new("TensorC*"); $Vecs = $ffi->new("TensorC*");
        $ffi->tensor_eigen_sym($this->ptr, \FFI::addr($Vals), \FFI::addr($Vecs));
        self::checkError();
        return ['values' => self::wrap($Vals), 'vectors' => self::wrap($Vecs)];
    }
    
    public function ref(): self { $res = self::wrap(TensorEngine::get()->tensor_ref($this->ptr)); self::checkError(); return $res; }
    public function rref(): self { $res = self::wrap(TensorEngine::get()->tensor_rref($this->ptr)); self::checkError(); return $res; }

    // --- DEEP LEARNING (CNN Primitives) ---
    public function im2col(int $kh, int $kw, int $sh = 1, int $sw = 1, int $ph = 0, int $pw = 0): self {
        $res = self::wrap(TensorEngine::get()->tensor_im2col($this->ptr, $kh, $kw, $sh, $sw, $ph, $pw));
        self::checkError(); return $res;
    }
    public function col2im(int $b, int $c, int $h, int $w, int $kh, int $kw, int $sh = 1, int $sw = 1, int $ph = 0, int $pw = 0): self {
        $res = self::wrap(TensorEngine::get()->tensor_col2im($this->ptr, $b, $c, $h, $w, $kh, $kw, $sh, $sw, $ph, $pw));
        self::checkError(); return $res;
    }
    public function conv2d(Tensor $W, ?Tensor $bias = null, int $sh = 1, int $sw = 1, int $ph = 0, int $pw = 0): self {
        $b_ptr = $bias?->ptr;
        $res = self::wrap(TensorEngine::get()->tensor_conv2d($this->ptr, $W->ptr, $b_ptr, $sh, $sw, $ph, $pw));
        self::checkError(); return $res;
    }
    public function conv2dBackward(Tensor $X, Tensor $W, int $sh = 1, int $sw = 1, int $ph = 0, int $pw = 0): array {
        $ffi = TensorEngine::get();
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
        $res = self::wrap(TensorEngine::get()->tensor_embedding_lookup($this->ptr, $weights->ptr));
        self::checkError(); return $res;
    }

    // --- I/O SERIALIZATION ---
    public function save(string $filepath): void { 
        TensorEngine::get()->tensor_save_to_file($this->ptr, $filepath); 
        self::checkError(); 
    }
    public static function load(string $filepath): self { 
        $res = self::wrap(TensorEngine::get()->tensor_load_from_file($filepath)); 
        self::checkError(); return $res; 
    }

    // --- DATA INGESTION ---

    /**
     * Converts a nested PHP array into a contiguous C Tensor.
     * @param array $data Deeply nested array of numbers
     * @param int $dtype defaults to Tensor::DTYPE_FLOAT32
     */
    public static function fromArray(array $data, int $dtype = self::DTYPE_FLOAT32): self {
        if (empty($data)) {
            throw new \InvalidArgumentException("Cannot create Tensor from empty array.");
        }

        $shape = [];
        $current = $data;
        while (\is_array($current)) {
            $shape[] = \count($current);
            if (empty($current)) break;
            $current = $current[\array_key_first($current)];
        }

        $tensor = new self($shape, $dtype); 
        $expectedSize = $tensor->size();

        $i = 0;
        self::flattenWriteJIT($data, $tensor->buffer, $i, $expectedSize, $dtype);

        if ($i !== $expectedSize) {
            throw new \InvalidArgumentException("Jagged array detected. Tensors must be perfectly rectangular.");
        }

        return $tensor;
    }

    private static function flattenWriteJIT(array $data, \FFI\CData $cdata, int &$i, int $expectedSize, int $dtype): void 
    {
        $isInt = $dtype === self::DTYPE_INT32 || $dtype === self::DTYPE_INT64;
        
        foreach ($data as $val) {
            if (\is_array($val)) {
                self::flattenWriteJIT($val, $cdata, $i, $expectedSize, $dtype);
            } else {
                if ($i >= $expectedSize) {
                    throw new \InvalidArgumentException("Jagged array detected. Inner dimensions exceed the inferred shape.");
                }
                
                if ($isInt) {
                    $cdata[$i++] = (int) $val;
                } else {
                    $cdata[$i++] = (float) $val;
                }
            }
        }
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

        if ($dtype === self::DTYPE_INT32) {
            $unpacked = \unpack('l*', $binary);
        } elseif ($dtype === self::DTYPE_INT64) {
            $unpacked = \unpack('q*', $binary);
        } else {
            $unpacked = \unpack('f*', $binary);
        }

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
    
    public function __destruct()
    {
        if ($this->owned && $this->ptr !== null) {
            // Memory safe call: if this tensor was born in an Arena, 
            // tensor_free safely intercepts it and does nothing!
            $ffi = TensorEngine::get();
            $ffi->tensor_free($this->ptr); // @phpstan-ignore-line — FFI methods are resolved at runtime
            $this->ptr = null;
            $this->buffer = null;
        }
    }
}