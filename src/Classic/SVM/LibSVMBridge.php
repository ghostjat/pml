<?php

declare(strict_types=1);

namespace Pml\Classic\SVM;

// ═══════════════════════════════════════════════════════════════════════════
//  LibSVMBridge — FFI singleton for libsvm.so
//
//  Provides the C type definitions and function bindings needed to call
//  libsvm's training and prediction routines from PHP.
//
//  ── libsvm Data Layout ───────────────────────────────────────────────────
//
//  libsvm uses a SPARSE feature representation: every sample is an array of
//  (index, value) pairs, terminated by a sentinel node with index = -1.
//
//    svm_node nodes[d+1];        // d features, 1 sentinel
//    nodes[j] = { j+1, x[j] };  // 1-indexed feature indices (libsvm convention)
//    nodes[d] = { -1,  0.0 };   // sentinel: marks end of sparse vector
//
//  For dense Pml Tensors we always generate the full d-feature representation
//  (no sparsity), so every sample produces exactly d+1 svm_nodes.
//
//  ── Memory Ownership ─────────────────────────────────────────────────────
//
//  All svm_node arrays and the svm_problem struct are allocated via
//  $ffi->new(...) with PHP GC ownership (owned=true, the default).
//  They are only needed DURING the svm_train() call — libsvm copies all
//  data into its internal model representation.  After svm_train() returns,
//  PHP's GC can free these temporary structures.
//
//  The returned svm_model* is owned by C (libsvm allocated it via malloc).
//  It MUST be freed with svm_free_and_destroy_model() — never PHP's GC.
//  SVC and SVR store the model pointer in a void*[1] "box" so that
//  \FFI::addr($box[0]) yields a void** for the free call.
//
//  ── SVM Types ────────────────────────────────────────────────────────────
//
//    C_SVC       (0) — C-Support Vector Classification
//    NU_SVC      (1) — ν-Support Vector Classification
//    ONE_CLASS   (2) — One-class SVM (anomaly detection)
//    EPSILON_SVR (3) — ε-Support Vector Regression
//    NU_SVR      (4) — ν-Support Vector Regression
//
//  ── Kernel Types ─────────────────────────────────────────────────────────
//
//    LINEAR      (0) — u^T v
//    POLY        (1) — (γ u^T v + r)^degree
//    RBF         (2) — exp(−γ ||u−v||²)  ← default
//    SIGMOID     (3) — tanh(γ u^T v + r)
//    PRECOMPUTED (4) — precomputed kernel matrix
// ═══════════════════════════════════════════════════════════════════════════

final class LibSVMBridge
{
    private static ?self $instance = null;

    /** The loaded FFI object — call BLAS-style functions directly on this. */
    public readonly \FFI $ffi;

    // ── FFI Header ────────────────────────────────────────────────────────

    private const HEADER = <<<'C'
        /* ── svm_node: one (feature-index, feature-value) pair ─────────────────
         *
         *  libsvm uses SPARSE input: each sample is an array of svm_node structs
         *  ending with a sentinel node whose index = -1.
         *
         *  For dense data (Pml Tensors), every feature is always present:
         *    node[j] = { index = j+1,  value = X[i,j] }   j ∈ [0, d)
         *    node[d] = { index = -1,   value = 0.0 }       sentinel
         *
         *  IMPORTANT: index is 1-based (libsvm convention), and value uses
         *  C double (64-bit), NOT float.  We widen from float32 on marshalling.
         */
        typedef struct {
            int    index;
            double value;
        } svm_node;

        /* ── svm_parameter: hyperparameter bundle ───────────────────────────────
         *
         *  Passed by pointer to svm_train().  libsvm reads — but does not own —
         *  this struct, so it is safe to allocate it on the PHP stack (via
         *  $ffi->new("svm_parameter")) and let PHP free it after svm_train.
         *
         *  For per-class weights (nr_weight > 0): weight_label and weight must
         *  point to matching arrays of length nr_weight.
         *  When nr_weight = 0 (the common case), set both pointers to NULL.
         */
        typedef struct {
            int     svm_type;      /* 0=C_SVC 1=NU_SVC 2=ONE_CLASS 3=EPS_SVR 4=NU_SVR */
            int     kernel_type;   /* 0=LINEAR 1=POLY 2=RBF 3=SIGMOID 4=PRECOMPUTED */
            int     degree;        /* POLY: (γ·u^Tv + coef0)^degree */
            double  gamma;         /* RBF/POLY/SIGMOID kernel coefficient */
            double  coef0;         /* POLY/SIGMOID independent term */
            double  cache_size;    /* kernel cache size in MB */
            double  eps;           /* stopping tolerance (default 1e-3) */
            double  C;             /* regularisation parameter for SVC/SVR */
            int     nr_weight;     /* number of per-class weight overrides (0 = none) */
            int    *weight_label;  /* class labels for per-class weights (NULL when nr_weight=0) */
            double *weight;        /* per-class C multipliers (NULL when nr_weight=0) */
            double  nu;            /* NU_SVC/NU_SVR/ONE_CLASS: nu ∈ (0,1] */
            double  p;             /* EPSILON_SVR: ε-tube half-width */
            int     shrinking;     /* 1 = use shrinking heuristics */
            int     probability;   /* 1 = enable probability estimates */
        } svm_parameter;

        /* ── svm_problem: the training dataset ──────────────────────────────────
         *
         *  l   = number of training samples
         *  y   = label array, double[l]         (regression: continuous values)
         *  x   = feature array, svm_node*[l]    (pointer to each sample's nodes)
         *
         *  svm_train() reads this struct and copies all necessary data into the
         *  returned svm_model, so problem storage can be freed immediately after.
         */
        typedef struct {
            int       l;
            double   *y;
            svm_node **x;
        } svm_problem;

        /* ── svm_model: opaque C-heap handle ────────────────────────────────────
         *
         *  We never dereference svm_model from PHP — only store/pass the pointer.
         *  Using void* avoids needing the exact struct layout.
         *  The actual libsvm C functions accept struct svm_model* which is
         *  ABI-equivalent to void* on all supported platforms.
         *
         *  Lifecycle:
         *    svm_train()                 → allocates the model (C malloc, not PHP GC)
         *    svm_predict()               → read-only access
         *    svm_free_and_destroy_model  → frees the model AND nulls the pointer
         */
        void  *svm_train(const svm_problem *prob, const svm_parameter *param);
        double svm_predict(const void *model, const svm_node *x);
        void   svm_free_and_destroy_model(void **model_ptr_ptr);

        /* ── Diagnostics ─────────────────────────────────────────────────────── */
        int  svm_check_parameter(const svm_problem *prob, const svm_parameter *param);
        int  svm_get_nr_class(const void *model);
        void svm_get_labels(const void *model, int *label);
    C;

    // ── Library search paths ──────────────────────────────────────────────

    private const LIB_CANDIDATES = [
        '/usr/lib/x86_64-linux-gnu/libsvm.so.3',
        '/usr/lib/x86_64-linux-gnu/libsvm.so',
        '/usr/lib/aarch64-linux-gnu/libsvm.so.3',
        '/usr/lib/aarch64-linux-gnu/libsvm.so',
        '/usr/local/lib/libsvm.so.3',
        '/usr/local/lib/libsvm.so',
        '/opt/homebrew/lib/libsvm.so',
        '/opt/homebrew/lib/libsvm.dylib',
        'libsvm.so.3',
        'libsvm.so',
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
                "Pml\Classic\SVM could not load libsvm.\n"
                . "Install it with: apt-get install libsvm-dev      (Debian/Ubuntu)\n"
                . "                  brew install libsvm              (macOS)\n"
                . "Searched: " . implode(', ', self::LIB_CANDIDATES)
            );
        }

        $this->ffi = $ffi;
    }

    public static function get(): self
    {
        return self::$instance ??= new self();
    }

    // ── SVM type constants (matches libsvm's svm.h) ───────────────────────

    public const C_SVC       = 0;
    public const NU_SVC      = 1;
    public const ONE_CLASS   = 2;
    public const EPSILON_SVR = 3;
    public const NU_SVR      = 4;

    // ── Kernel type constants ─────────────────────────────────────────────

    public const LINEAR      = 0;
    public const POLY        = 1;
    public const RBF         = 2;
    public const SIGMOID     = 3;
    public const PRECOMPUTED = 4;

    // ── Kernel name → constant map ────────────────────────────────────────

    public const KERNEL_MAP = [
        'linear'      => self::LINEAR,
        'poly'        => self::POLY,
        'polynomial'  => self::POLY,
        'rbf'         => self::RBF,
        'sigmoid'     => self::SIGMOID,
        'precomputed' => self::PRECOMPUTED,
    ];
}
