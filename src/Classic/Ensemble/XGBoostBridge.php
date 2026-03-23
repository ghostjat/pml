<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

// ═══════════════════════════════════════════════════════════════════════════
//  XGBoostBridge — FFI singleton for libxgboost.so
//
//  Provides the C API bindings for XGBoost's C interface.  All XGBoost
//  objects (DMatrix, Booster) are opaque void* handles managed by XGBoost's
//  own allocator — PHP's GC knows nothing about them.  Callers MUST call
//  XGDMatrixFree / XGBoosterFree (or let __destruct do it).
//
//  ── Handle Types ─────────────────────────────────────────────────────────
//
//  The XGBoost C API uses two opaque handle types:
//
//    DMatrixHandle  — a pointer to XGBoost's internal dense/sparse matrix.
//                     Created from raw float* data via XGDMatrixCreateFromMat.
//                     Freed with XGDMatrixFree().
//
//    BoosterHandle  — a pointer to a trained XGBoost model (the gradient
//                     boosting ensemble).
//                     Created with XGBoosterCreate() from one or more DMatrices.
//                     Freed with XGBoosterFree().
//
//  Both are typedef'd to void* in xgboost/c_api.h.  We keep them as void*
//  throughout to avoid needing struct definitions.
//
//  ── Zero-Copy DMatrix Creation ───────────────────────────────────────────
//
//  XGDMatrixCreateFromMat accepts a const float* pointing to a row-major
//  dense matrix.  Since Pml Tensors store data in a flat float[n*d] FFI
//  buffer, we pass $X->buffer directly — no copy, no temporary allocation.
//  XGBoost reads and copies the data into its own compressed representation
//  during the call, so our buffer can be freed or reused immediately.
//
//  ── Prediction Output ────────────────────────────────────────────────────
//
//  XGBoosterPredict writes a const float* pointing to an XGBoost-internal
//  buffer into out_result.  This buffer is valid until the next prediction
//  call or until the Booster is freed.  We immediately copy it into a new
//  Pml Tensor using \FFI::memcpy to avoid dangling pointer issues.
//
//  ── Error Handling ───────────────────────────────────────────────────────
//
//  All XGBoost C API functions return int:
//    0 → success
//    1 → error (call XGBGetLastError() for the message)
//
//  XGBClassifier checks every return code and throws on failure.
// ═══════════════════════════════════════════════════════════════════════════

final class XGBoostBridge
{
    private static ?self $instance = null;

    /** The loaded FFI object. */
    public readonly \FFI $ffi;

    // ── FFI Header ────────────────────────────────────────────────────────

    private const HEADER = <<<'C'
        /* ── XGBoost C API header subset ────────────────────────────────────────
         *
         *  All handles (DMatrixHandle, BoosterHandle) are void* in xgboost/c_api.h.
         *  Using void* here avoids forward-declaring opaque XGBoost structs and
         *  is ABI-identical on all supported platforms.
         *
         *  bst_ulong = uint64_t on all XGBoost builds.  We typedef it explicitly
         *  because the FFI header scope does not inherit system typedefs.
         */
        typedef unsigned long long bst_ulong;

        /* ── DMatrix creation ────────────────────────────────────────────────────
         *
         *  XGDMatrixCreateFromMat: build a DMatrix from a row-major dense matrix.
         *
         *  data    — pointer to float32 row-major [nrow × ncol] matrix
         *  nrow    — number of samples
         *  ncol    — number of features
         *  missing — value that marks a missing entry (use NaN for "no missing")
         *  out     — output: void** filled with the new DMatrixHandle
         *
         *  XGBoost copies data internally; the float* buffer can be freed
         *  immediately after this call returns.
         */
        int XGDMatrixCreateFromMat(const float *data, bst_ulong nrow, bst_ulong ncol,
                                   float missing, void **out);

        /* ── DMatrix label / weight binding ─────────────────────────────────────
         *
         *  XGDMatrixSetFloatInfo: attach metadata to a DMatrix.
         *
         *  field = "label"  → target values for training  (float array, length nrow)
         *  field = "weight" → per-sample importance weights
         *
         *  XGBoost copies the array; our float* buffer can be freed afterwards.
         */
        int XGDMatrixSetFloatInfo(void *handle, const char *field,
                                  const float *array, bst_ulong len);

        /* ── Booster lifecycle ───────────────────────────────────────────────────
         *
         *  XGBoosterCreate: create a booster associated with one or more DMatrices.
         *    dmats — array of DMatrixHandle (void*[]), length len
         *    len   — number of DMatrices (usually 1 = training set)
         *    out   — output: void** filled with the new BoosterHandle
         */
        int XGBoosterCreate(const void **dmats, bst_ulong len, void **out);

        /* ── Booster parameter setting ───────────────────────────────────────────
         *
         *  XGBoosterSetParam: set a key-value parameter on the booster.
         *  All XGBoost params are strings (name=value pairs).
         *  Must be called BEFORE the first XGBoosterUpdateOneIter.
         *
         *  Common params:
         *    "max_depth"      → tree depth (int)
         *    "eta"            → learning rate (float)
         *    "objective"      → "binary:logistic" | "multi:softmax" | "reg:squarederror"
         *    "num_class"      → required for multi:softmax
         *    "eval_metric"    → "logloss" | "mlogloss" | "rmse"
         *    "subsample"      → row subsampling ratio
         *    "colsample_bytree" → column subsampling ratio
         */
        int XGBoosterSetParam(void *handle, const char *name, const char *value);

        /* ── Boosting loop ───────────────────────────────────────────────────────
         *
         *  XGBoosterUpdateOneIter: run one boosting round (add one tree or set
         *  of trees).  Call in a loop for n_estimators rounds.
         *
         *  iter  — current iteration index (0-based), used for learning rate decay
         *  dtrain — the training DMatrix
         */
        int XGBoosterUpdateOneIter(void *handle, int iter, void *dtrain);

        /* ── Prediction ──────────────────────────────────────────────────────────
         *
         *  XGBoosterPredict: run inference on a DMatrix.
         *
         *  option_mask  — 0 = normal prediction; 1 = output margin (pre-sigmoid)
         *  ntree_limit  — 0 = use all trees; N = use first N trees
         *  training     — 0 = inference mode (disables dropout etc.)
         *  out_len      — output: number of floats in out_result
         *  out_result   — output: float** — XGBoost's internal buffer
         *                 VALID only until next predict call or XGBoosterFree.
         *                 Copy immediately with memcpy.
         *
         *  Output layout:
         *    binary:logistic → out_len = n_samples, one sigmoid prob per sample
         *    multi:softmax   → out_len = n_samples, one class index per sample
         *    multi:softprob  → out_len = n_samples * n_classes, row-major probs
         */
        int XGBoosterPredict(void *handle, void *dmat, int option_mask,
                             unsigned int ntree_limit, int training,
                             bst_ulong *out_len, const float **out_result);

        /* ── Cleanup ─────────────────────────────────────────────────────────── */
        int XGDMatrixFree(void *handle);
        int XGBoosterFree(void *handle);

        /* ── Error reporting ─────────────────────────────────────────────────── */
        const char *XGBGetLastError(void);
    C;

    // ── Library search paths ──────────────────────────────────────────────

    private const LIB_CANDIDATES = [
        '/usr/lib/x86_64-linux-gnu/libxgboost.so',
        '/usr/lib/aarch64-linux-gnu/libxgboost.so',
        '/usr/local/lib/libxgboost.so',
        '/opt/homebrew/lib/libxgboost.dylib',
        '/opt/homebrew/lib/libxgboost.so',
        'libxgboost.so',
        'xgboost.so',
    ];

    private function __construct()
    {
        $ffi = null;
        foreach (self::LIB_CANDIDATES as $lib) {
            try {
                $ffi = \FFI::cdef(self::HEADER, $lib);
                break;
            } catch (\FFI\Exception) {
                continue;
            }
        }

        if ($ffi === null) {
            throw new \RuntimeException(
                "Pml\Classic\Ensemble could not load libxgboost.\n"
                . "Install it with: pip install xgboost (then locate libxgboost.so)\n"
                . "              or: apt-get install libxgboost-dev  (if packaged)\n"
                . "              or: brew install xgboost             (macOS)\n"
                . "Searched: " . implode(', ', self::LIB_CANDIDATES)
            );
        }

        $this->ffi = $ffi;
    }

    public static function get(): self
    {
        return self::$instance ??= new self();
    }

    /**
     * Check an XGBoost C API return code and throw on error.
     *
     * Every XGBoost C function returns 0 on success, 1 on failure.
     * On failure, XGBGetLastError() returns a static C string with details.
     */
    public function check(int $ret, string $fn): void
    {
        if ($ret !== 0) {
            $msg = \FFI::string($this->ffi->XGBGetLastError());
            throw new \RuntimeException("XGBoost {$fn} failed: {$msg}");
        }
    }
}
