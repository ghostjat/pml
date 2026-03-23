<?php

declare(strict_types=1);

namespace Pml\Classic\Utils;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  Joblib — FFI-safe model serialisation (mirrors joblib.dump / joblib.load)
//
//  PHP's standard serialize() / unserialize() is INCOMPATIBLE with objects
//  that contain \FFI\CData buffers (Tensor::$buffer, Tensor::$grad).
//  Attempting to serialize such objects causes a fatal error at the CData
//  boundary.
//
//  This class implements a two-phase approach:
//
//  ── dump() — FFI-Safe Serialisation ─────────────────────────────────────
//
//  1. REFLECTION TRAVERSAL: Walk the model object's property tree using
//     ReflectionClass / ReflectionProperty.
//
//  2. TENSOR EXTRACTION: For every Tensor encountered, extract its raw
//     bytes from C memory using \FFI::string($buf, size * 4).  Store the
//     bytes + shape + dtype in a plain PHP surrogate array marked with
//     '__pml_tensor__' => true.
//
//  3. OBJECT SURROGATES: For non-Tensor objects (nested estimators,
//     pipeline steps), recursively build a surrogate array marked with
//     '__pml_object__' => true, containing the class name and all property
//     surrogates.
//
//  4. SAFE SERIALIZE: The entire surrogate tree (plain PHP scalars, arrays,
//     and surrogate arrays) is serialized with PHP's serialize() and written
//     to disk.  No CData values remain in the serialized data.
//
//  ── load() — FFI Reconstruction ─────────────────────────────────────────
//
//  1. Unserialize the file's byte string back to PHP arrays/scalars.
//
//  2. TENSOR RECONSTRUCTION: For each '__pml_tensor__' surrogate:
//       a. Allocate a new C memory buffer via BlasEngine::get()->allocFloat().
//       b. Copy the saved binary string back into C memory via \FFI::memcpy().
//       c. Construct a new Tensor with the pre-populated buffer.
//       d. Restore mutable metadata (_transposed, requiresGrad, etc.).
//
//  3. OBJECT RECONSTRUCTION: For each '__pml_object__' surrogate:
//       a. ReflectionClass::newInstanceWithoutConstructor() — creates a blank
//          instance with NO constructor execution.  This leaves all readonly
//          properties UNINITIALIZED, which is the only state from which PHP 8.1
//          allows ReflectionProperty::setValue() to initialize them.
//       b. For each stored property, recursively reconstruct the value and
//          inject it via ReflectionProperty::setValue().
//
//  ── What is and is NOT serialised ────────────────────────────────────────
//
//  Serialised:
//    - Tensor buffers and metadata (shape, dtype, quantScale, etc.)
//    - Tensor::$grad buffer (if non-null — rare for Classic models)
//    - All scalar, int[], float[], string fitted attributes
//    - Nested estimator objects (Pipeline steps, RandomForest::$estimators_)
//    - PHP arrays (flat arrays, arrays of surrogates)
//
//  NOT serialised (set to safe defaults on load):
//    - \Closure properties ($tensor->_backward, autograd backward functions)
//      → set to null
//    - Tensor::$_prev (autograd parent list)
//      → stays [] (PHP class default)
//    - Uninitialised properties not stored in the surrogate
//      → skipped silently (checkFitted() guards will catch any misuse)
//
//  ── Thread Safety ────────────────────────────────────────────────────────
//
//  dump() and load() are stateless static methods.  PHP requests are
//  single-threaded; no locking is needed for typical CLI / FPM usage.
// ═══════════════════════════════════════════════════════════════════════════

final class Joblib
{
    // Sentinel keys used to tag surrogate arrays and distinguish them from
    // plain PHP arrays during deserialization.
    private const TAG_TENSOR = '__pml_tensor__';
    private const TAG_OBJECT = '__pml_object__';

    // ── Public API ─────────────────────────────────────────────────────────

    /**
     * Serialise a fitted model to a file.
     *
     * Any Tensor objects in the model's property tree are extracted to safe
     * PHP surrogates before serialisation.  The resulting file contains only
     * standard PHP-serializable data.
     *
     * @param object $model     Fitted Pml\Classic estimator (or any object
     *                          whose property tree may contain Tensors).
     * @param string $filename  Destination file path.
     *
     * @throws \RuntimeException If the file cannot be written.
     */
    public static function dump(object $model, string $filename): void
    {
        // Walk the object tree and replace all CData/Tensor values with safe surrogates.
        $surrogate = self::serializeValue($model);

        // PHP serialize() produces a binary string of the safe surrogate tree.
        $bytes = serialize($surrogate);

        if (file_put_contents($filename, $bytes, LOCK_EX) === false) {
            throw new \RuntimeException(
                "Joblib::dump() failed to write '{$filename}'. "
                . 'Check directory permissions.'
            );
        }
    }

    /**
     * Load and reconstruct a fitted model from a file written by dump().
     *
     * Tensor buffers are reallocated in C memory and populated from the stored
     * binary data.  The returned object is fully functional.
     *
     * @param string $filename  Path to the file created by dump().
     * @return object           Reconstructed fitted model.
     *
     * @throws \RuntimeException   If the file cannot be read.
     * @throws \UnexpectedValueException  If the file does not contain a valid surrogate.
     */
    public static function load(string $filename): object
    {
        $bytes = file_get_contents($filename);
        if ($bytes === false) {
            throw new \RuntimeException(
                "Joblib::load() failed to read '{$filename}'. Does the file exist?"
            );
        }

        // Unserialize the safe surrogate tree (no CData inside — PHP handles this fine).
        $surrogate = unserialize($bytes, ['allowed_classes' => false]);

        if ($surrogate === false) {
            throw new \UnexpectedValueException(
                "Joblib::load(): failed to unserialize '{$filename}'. The file may be corrupted."
            );
        }

        // Reconstruct all Tensors and nested objects from surrogates.
        $value = self::deserializeValue($surrogate);

        if (!is_object($value)) {
            throw new \UnexpectedValueException(
                "Joblib::load(): top-level surrogate is not an object. "
                . 'Was this file written by Joblib::dump()?'
            );
        }

        return $value;
    }

    // ── Serialisation (object → safe surrogate) ────────────────────────────

    /**
     * Recursively convert a PHP value to a CData-free surrogate.
     *
     * Decision tree:
     *   Tensor    → '__pml_tensor__' surrogate array (binary data extracted)
     *   \Closure  → null             (not serialisable; safe default on reload)
     *   object    → '__pml_object__' surrogate array (properties recursed)
     *   array     → element-wise recursion
     *   scalar/null → returned as-is
     */
    private static function serializeValue(mixed $value): mixed
    {
        // ── Tensor: extract raw C memory to a PHP binary string ─────────────
        if ($value instanceof Tensor) {
            return self::serializeTensor($value);
        }

        // ── Closure: cannot be serialised; becomes null on restore ───────────
        if ($value instanceof \Closure) {
            return null;
        }

        // ── General object: build an '__pml_object__' surrogate ──────────────
        if (is_object($value)) {
            return self::serializeObject($value);
        }

        // ── Array: recurse into each element ─────────────────────────────────
        if (is_array($value)) {
            $result = [];
            foreach ($value as $k => $v) {
                $result[$k] = self::serializeValue($v);
            }
            return $result;
        }

        // ── Scalar (bool, int, float, string) or null: safe as-is ────────────
        return $value;
    }

    /**
     * Build a Tensor surrogate.
     *
     * The surrogate contains:
     *   - shape, size, strides, dtype, quantScale, quantZeroPoint  (metadata)
     *   - binary_data  : raw bytes from the float[N] C buffer as a PHP string
     *   - grad_data    : same for the gradient buffer, or null
     *   - _transposed, _transposedShape, requiresGrad              (mutable flags)
     *
     * Binary extraction:
     *   \FFI::string($tensor->buffer, $byteCount)
     *   reads $byteCount bytes starting at the buffer pointer into a PHP string.
     *   For FLOAT32: byteCount = size * 4  (4 bytes per float32)
     *   For INT8:    byteCount = size * 1
     */
    private static function serializeTensor(Tensor $t): array
    {
        $bytesPerElement = ($t->dtype === Tensor::INT8) ? 1 : 4;

        // ── Extract raw C memory into a PHP binary string ──────────────────
        // \FFI::string() reads bytes directly from the C buffer pointer.
        $binaryData = \FFI::string($t->buffer, $t->size * $bytesPerElement);

        $surrogate = [
            self::TAG_TENSOR     => true,
            'shape'              => $t->shape,
            'size'               => $t->size,
            'strides'            => $t->strides,
            'dtype'              => $t->dtype,
            'quantScale'         => $t->quantScale,
            'quantZeroPoint'     => $t->quantZeroPoint,
            '_transposed'        => $t->_transposed,
            '_transposedShape'   => $t->_transposedShape,
            'requiresGrad'       => $t->requiresGrad,
            'binary_data'        => $binaryData,
            'grad_data'          => null,
        ];

        // ── Optionally serialise the gradient buffer ───────────────────────
        // For Classic models this is almost always null (no autograd during inference).
        // We still support it for completeness.
        if ($t->grad !== null) {
            // Gradient is always float32, same size as the primary buffer
            $surrogate['grad_data'] = \FFI::string($t->grad, $t->size * 4);
        }

        return $surrogate;
    }

    /**
     * Build an object surrogate using Reflection.
     *
     * Steps:
     *  1. ReflectionClass::getProperties() with ALL visibility flags —
     *     this includes private, protected, and inherited properties.
     *     getProperties() without flags returns public only.
     *
     *  2. For each property:
     *     a. setAccessible(true) — lifts visibility restriction for getValue().
     *     b. isInitialized($obj) — guards against reading an uninitialized
     *        readonly property (throws Error otherwise).
     *     c. getValue($obj) is recursively serialised.
     *
     *  3. \Closure values (autograd $_backward) are converted to null by
     *     serializeValue() — they are not stored in __props__.
     *
     *  The '__class__' key stores the fully-qualified class name so
     *  deserializeObject() can call newInstanceWithoutConstructor() on the
     *  right class.
     */
    private static function serializeObject(object $obj): array
    {
        $rc    = new \ReflectionClass($obj);
        $props = [];

        // getProperties() without a filter flag returns only public properties.
        // We need ALL properties including private/protected ones.
        // ReflectionClass::getProperties(ReflectionProperty::IS_*) can be ORed.
        $allProps = $rc->getProperties(
            \ReflectionProperty::IS_PUBLIC
            | \ReflectionProperty::IS_PROTECTED
            | \ReflectionProperty::IS_PRIVATE
        );

        foreach ($allProps as $prop) {
            // Allow access to private/protected properties
            $prop->setAccessible(true);

            // ── Guard: uninitialized readonly properties ───────────────────
            // If a readonly property (e.g. coef_) was never assigned because
            // the model has not been fully fitted, isInitialized() returns false.
            // Reading it would throw a PHP Error — skip it instead.
            // On load, the missing key simply means the property stays unset,
            // which is the correct "not yet fitted" state.
            if (!$prop->isInitialized($obj)) {
                continue;
            }

            $name = $prop->getName();
            $raw  = $prop->getValue($obj);

            // serializeValue handles Closures (→ null), Tensors, nested objects, arrays.
            $serialized = self::serializeValue($raw);

            // Store nullified Closures with their property name so that
            // deserializeObject() can set them back to null explicitly.
            // (The _backward / _prev case — important for correctness.)
            $props[$name] = $serialized;
        }

        return [
            self::TAG_OBJECT => true,
            '__class__'      => get_class($obj),   // fully-qualified class name
            '__props__'      => $props,
        ];
    }

    // ── Deserialisation (safe surrogate → live object) ─────────────────────

    /**
     * Recursively reconstruct a value from its surrogate.
     *
     * Tagged arrays (__pml_tensor__, __pml_object__) are dispatched to their
     * specific reconstructors.  Plain arrays are element-wise reconstructed.
     * Scalars and null are returned directly.
     */
    private static function deserializeValue(mixed $value): mixed
    {
        // Only arrays carry surrogate tags — scalars/null are returned as-is.
        if (!is_array($value)) {
            return $value;
        }

        // ── Tensor surrogate ─────────────────────────────────────────────────
        if (isset($value[self::TAG_TENSOR])) {
            return self::deserializeTensor($value);
        }

        // ── Object surrogate ─────────────────────────────────────────────────
        if (isset($value[self::TAG_OBJECT])) {
            return self::deserializeObject($value);
        }

        // ── Plain array: recurse element-wise ─────────────────────────────────
        $result = [];
        foreach ($value as $k => $v) {
            $result[$k] = self::deserializeValue($v);
        }
        return $result;
    }

    /**
     * Reconstruct a Tensor from its surrogate.
     *
     * Steps:
     *  1. Allocate a new C memory buffer of the appropriate size and dtype.
     *  2. Copy the binary PHP string back into C memory via \FFI::memcpy().
     *     This avoids any element-wise PHP loop — one C-level memcpy call
     *     copies all N * byteSize bytes.
     *  3. Construct the Tensor with the pre-populated buffer passed as the
     *     $buffer argument to the constructor (prevents double-allocation).
     *  4. Restore mutable metadata directly on public properties.
     *  5. Reconstruct the gradient buffer if it was serialised.
     */
    private static function deserializeTensor(array $s): Tensor
    {
        $dtype           = $s['dtype'];
        $size            = $s['size'];
        $bytesPerElement = ($dtype === Tensor::INT8) ? 1 : 4;
        $byteCount       = $size * $bytesPerElement;

        // ── Step 1: Allocate a new C memory buffer ─────────────────────────
        // The second argument (true) means GC-owned — PHP will free it when
        // the Tensor goes out of scope, matching the standard Tensor lifecycle.
        if ($dtype === Tensor::INT8) {
            $buf = BlasEngine::get()->allocInt8($size, true);
        } else {
            $buf = BlasEngine::get()->allocFloat($size, true);
        }

        // ── Step 2: Copy binary data from PHP string into C memory ─────────
        // \FFI::memcpy(\FFI\CData $dst, string|CData $src, int $size)
        // accepts a PHP string as $src — it reads bytes directly from the
        // PHP string's internal char* buffer.
        \FFI::memcpy($buf, $s['binary_data'], $byteCount);

        // ── Step 3: Construct the Tensor with the pre-populated buffer ──────
        // Passing $buf as the $buffer argument to the constructor causes the
        // Tensor to assign it directly (no new allocation occurs):
        //   if ($buffer !== null) { $this->buffer = $buffer; }
        $tensor = new Tensor(
            $s['shape'],
            $buf,
            $dtype,
            $s['quantScale'],
            $s['quantZeroPoint'],
        );

        // ── Step 4: Restore mutable metadata ─────────────────────────────────
        // These are public non-readonly properties initialised by class defaults,
        // so we simply overwrite them.
        $tensor->_transposed      = $s['_transposed'];
        $tensor->_transposedShape = $s['_transposedShape'];
        $tensor->requiresGrad     = $s['requiresGrad'];

        // ── Step 5: Reconstruct gradient buffer (if present) ─────────────────
        // For Classic models this will almost always be null.
        if ($s['grad_data'] !== null) {
            // initGrad() allocates a zeroed float[size] buffer for $tensor->grad.
            $tensor->initGrad();
            \FFI::memcpy($tensor->grad, $s['grad_data'], $size * 4);
        }

        return $tensor;
    }

    /**
     * Reconstruct an object from its surrogate using Reflection.
     *
     * ── Why newInstanceWithoutConstructor()? ──────────────────────────────
     *
     * PHP 8.1 readonly properties can only be initialised ONCE.  If the
     * constructor runs, it would try to assign them — but the constructor
     * arguments are hyperparameters, not fitted attributes.  Running it would
     * set some properties and leave others (like coef_) unset.  Then calling
     * ReflectionProperty::setValue() on an already-set readonly throws an Error.
     *
     * newInstanceWithoutConstructor() creates a blank instance where ALL
     * readonly properties are uninitialized.  We can then setValue() each one
     * exactly once, which is the only operation PHP 8.1 readonly semantics permit.
     *
     * This pattern is well-established: it is how Symfony Serializer,
     * Doctrine, and other frameworks handle readonly properties.
     *
     * ── Property injection order ──────────────────────────────────────────
     *
     * Properties are injected in the order they appear in '__props__', which
     * matches the order they were discovered by getProperties() during dump().
     * No ordering dependency exists between fitted attributes.
     *
     * ── Skipped properties ────────────────────────────────────────────────
     *
     * If a property name in '__props__' no longer exists in the class (due to
     * a code change after the model was saved), it is silently skipped.  This
     * provides forward-compatibility with minor model class refactors.
     */
    private static function deserializeObject(array $surrogate): object
    {
        $className = $surrogate['__class__'];
        $propsData = $surrogate['__props__'];

        $rc = new \ReflectionClass($className);

        // ── Create a blank instance WITHOUT calling the constructor ─────────
        // All properties (including readonly) are uninitialized at this point.
        // PHP class-level defaults (e.g. public bool $flag = false) ARE applied
        // for non-readonly properties.  Constructor-promoted readonly properties
        // remain uninitialized.
        $obj = $rc->newInstanceWithoutConstructor();

        // ── Inject each stored property value ─────────────────────────────
        foreach ($propsData as $name => $rawSurrogate) {
            // ── Locate the property via Reflection ─────────────────────────
            // The property may be declared on a parent class; we search up the
            // hierarchy using a try-catch on ReflectionException.
            try {
                $prop = self::findProperty($rc, $name);
            } catch (\ReflectionException) {
                // Property no longer exists in the class hierarchy — skip.
                // This allows graceful loading of models saved by older code.
                continue;
            }

            $prop->setAccessible(true);   // access private/protected

            // ── Guard: do not set an already-initialized readonly property ─
            // Normally none should be initialized (we used newInstanceWithoutConstructor),
            // but guard defensively in case of future changes.
            if ($prop->isReadOnly() && $prop->isInitialized($obj)) {
                continue;
            }

            // ── Recursively reconstruct the value and inject it ─────────────
            $restoredValue = self::deserializeValue($rawSurrogate);
            $prop->setValue($obj, $restoredValue);
        }

        return $obj;
    }

    /**
     * Find a named property anywhere in the class hierarchy.
     *
     * ReflectionClass::getProperty($name) only searches the declared class,
     * not its parents.  For inherited private properties (e.g. from a base class
     * or trait), we must walk up via getParentClass().
     *
     * @throws \ReflectionException  If the property is not found in any ancestor.
     */
    private static function findProperty(\ReflectionClass $rc, string $name): \ReflectionProperty
    {
        $current = $rc;
        while ($current !== false) {
            if ($current->hasProperty($name)) {
                return $current->getProperty($name);
            }
            $current = $current->getParentClass();
        }

        throw new \ReflectionException(
            "Property '{$name}' not found in class '{$rc->getName()}' or any ancestor."
        );
    }
}
