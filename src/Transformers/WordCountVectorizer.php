<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Dataset;
use Pml\Interfaces\Saveable;
use Pml\Tensor;
use Pml\Lib\TensorEngine;
use RuntimeException;
use FFI;

final class WordCountVectorizer implements Saveable
{
    private ?int $maxFeatures;
    private ?FFI\CData $vocabPtr = null;
    private int $vocabSize = 0;
    private bool $fitted = false;
    private ?int $fittedColumnIdx = null;
    private string $textColumn;
    private ?string $vocabFilePath = null;

    public function __construct(?int $maxFeatures = null, string $textColumn = 'text')
    {
        $this->maxFeatures = $maxFeatures;
        $this->textColumn = $textColumn;
    }

    /**
     * Fit on the configured text column of a Dataset in ETL mode.
     */
    public function fit(Dataset $data): void
    {
        if (!$data->isEtlMode()) {
            throw new RuntimeException(
                'Dataset must be in ETL mode (use Dataset::load()). ' .
                'Materialised Datasets cannot be fitted.'
            );
        }

        $dfPtr = $data->getDataFramePointer();
        $colIdx = $this->resolveColumnIndex($data, $this->textColumn);

        $schema = $data->schema();
        if ($schema[$colIdx]['dtype'] !== 2) { // 2 = STRING
            throw new RuntimeException("Column '{$schema[$colIdx]['name']}' is not a text column.");
        }

        $ffi = TensorEngine::get();
        $this->vocabPtr = $ffi->df_vocab_build(
            $dfPtr,
            $colIdx,
            $this->maxFeatures ?? 0
        );
        self::checkError();

        $this->vocabSize = $ffi->vocab_size($this->vocabPtr);
        $this->fittedColumnIdx = $colIdx;
        $this->fitted = true;
    }

    /**
     * Transform a Dataset into a document‑term matrix.
     */
    public function transform(Dataset $data): Dataset
    {
        if (!$this->fitted) {
            throw new RuntimeException('Vectorizer has not been fitted.');
        }
        if (!$data->isEtlMode()) {
            throw new RuntimeException('Dataset must be in ETL mode for transformation.');
        }

        $dfPtr = $data->getDataFramePointer();
        $colIdx = $this->fittedColumnIdx;
        $schema = $data->schema();
        if (!isset($schema[$colIdx]) || $schema[$colIdx]['dtype'] !== 2) {
            throw new RuntimeException('Dataset does not contain the fitted text column.');
        }

        $ffi = TensorEngine::get();
        $tensorPtr = $ffi->df_transform_bow($dfPtr, $colIdx, $this->vocabPtr);
        self::checkError();

        $samples = Tensor::wrap($tensorPtr);

        // Extract ONLY the label column — avoids calling materialize() which
        // would fail when the DataFrame still contains string (text) columns.
        $labels = $data->isLabeled() ? $data->extractLabelTensor() : null;

        return new Dataset($samples, $labels);
    }

    /**
     * Fit and transform in one call.
     */
    public function fitTransform(Dataset $data): Dataset
    {
        $this->fit($data);
        return $this->transform($data);
    }

    /** Return the raw C Vocab* pointer for passing to pipeline_create(). */
    public function vocabPtr(): ?\FFI\CData { return $this->vocabPtr; }

    public function fitted(): bool
    {
        return $this->fitted;
    }

    public function vocabSize(): int
    {
        return $this->vocabSize;
    }

    public function getVocabulary(): array
    {
        if (!$this->fitted) {
            return [];
        }
        throw new RuntimeException('Vocabulary export not yet implemented in C layer.');
    }

    private function resolveColumnIndex(Dataset $data, string $column): int
    {
        $schema = $data->schema();
        $names = array_column($schema, 'name');
        foreach ($schema as $idx => $col) {
            if ($col['name'] === $column) {
                return $idx;
            }
        }
        $available = implode("', '", $names);
        throw new RuntimeException("Column '{$column}' not found. Available columns: '{$available}'");
    }

    private static function checkError(): void
    {
        $ffi = TensorEngine::get();
        if ($ffi->tensor_check_error()) {
            $raw = $ffi->tensor_get_last_error();
            // PHP FFI may return const char* as either CData or a PHP string
            // depending on version; normalise to string either way.
            $msg = is_string($raw) ? $raw : \FFI::string($raw);
            $ffi->tensor_clear_error();
            throw new RuntimeException($msg);
        }
    }

    // -------------------------------------------------------------------------
    // Clone — deep copy that prevents double-free of shared FFI\CData pointer.
    //
    // PHP clone() is shallow: both original and clone would share the same
    // vocabPtr (FFI\CData).  When either object is destroyed, vocab_free()
    // would run on the shared pointer — the survivor would then have a
    // dangling pointer.
    //
    // Fix: __clone() serialises the vocab to a temp file and nulls vocabPtr on
    // the clone.  The clone's __serialize() picks up the temp file; the temp
    // file is cleaned up by the clone's __destruct() if not consumed earlier.
    // -------------------------------------------------------------------------

    public function __clone()
    {
        if ($this->vocabPtr !== null) {
            $this->vocabFilePath = tempnam(sys_get_temp_dir(), 'pml_vocab_');
            TensorEngine::get()->vocab_save($this->vocabPtr, $this->vocabFilePath);
            // Null the pointer so this clone does not double-free C memory
            // that the original object still owns.
            $this->vocabPtr = null;
        }
    }

    // -------------------------------------------------------------------------
    // Serialization — vocab binary is embedded inline as base64 so that there
    // is NO dependency on an external temp-file path surviving across processes.
    // -------------------------------------------------------------------------

    public function __serialize(): array
    {
        $vocabB64 = '';

        if ($this->vocabPtr !== null) {
            // Original (non-cloned) path: save to temp, read bytes, embed, clean up.
            $tmp = tempnam(sys_get_temp_dir(), 'pml_vocab_');
            TensorEngine::get()->vocab_save($this->vocabPtr, $tmp);
            $vocabB64 = base64_encode((string) file_get_contents($tmp));
            unlink($tmp);
        } elseif ($this->vocabFilePath !== null && file_exists($this->vocabFilePath)) {
            // Clone path: __clone() already wrote the file; read and embed it.
            $vocabB64 = base64_encode((string) file_get_contents($this->vocabFilePath));
        }

        return [
            'maxFeatures'     => $this->maxFeatures,
            'textColumn'      => $this->textColumn,
            'vocabSize'       => $this->vocabSize,
            'fitted'          => $this->fitted,
            'fittedColumnIdx' => $this->fittedColumnIdx,
            'vocabB64'        => $vocabB64,
        ];
    }

    public function __unserialize(array $data): void
    {
        $this->maxFeatures     = $data['maxFeatures'];
        $this->textColumn      = $data['textColumn'];
        $this->vocabSize       = $data['vocabSize'];
        $this->fitted          = $data['fitted'];
        $this->fittedColumnIdx = $data['fittedColumnIdx'];
        $this->vocabFilePath   = null;
        $this->vocabPtr        = null;

        if (!empty($data['vocabB64'])) {
            // Inline binary — write to temp file, load, delete.
            $tmp = tempnam(sys_get_temp_dir(), 'pml_vocab_');
            file_put_contents($tmp, base64_decode($data['vocabB64']));
            $this->vocabPtr = TensorEngine::get()->vocab_load($tmp);
            unlink($tmp);
        } elseif (!empty($data['vocabFilePath']) && file_exists($data['vocabFilePath'])) {
            // Legacy fallback for models saved with the old temp-file format.
            $this->vocabPtr = TensorEngine::get()->vocab_load($data['vocabFilePath']);
            unlink($data['vocabFilePath']);
        }
    }

    // ── Saveable ──────────────────────────────────────────────────────────────
    // ModelStore uses these instead of Reflection so the FFI\CData $vocabPtr
    // is safely encoded as a base64 blob rather than silently skipped.

    public function getConfig(): array
    {
        return [
            'maxFeatures' => $this->maxFeatures,
            'textColumn'  => $this->textColumn,
        ];
    }

    public static function fromConfig(array $config): static
    {
        return new static($config['maxFeatures'] ?? null, $config['textColumn'] ?? 'text');
    }

    public function getPhpState(): array
    {
        $vocabB64 = '';
        if ($this->vocabPtr !== null) {
            $tmp = tempnam(\sys_get_temp_dir(), 'pml_vocab_');
            TensorEngine::get()->vocab_save($this->vocabPtr, $tmp);
            $vocabB64 = \base64_encode((string) \file_get_contents($tmp));
            \unlink($tmp);
        } elseif ($this->vocabFilePath !== null && \file_exists($this->vocabFilePath)) {
            $vocabB64 = \base64_encode((string) \file_get_contents($this->vocabFilePath));
        }

        return [
            'vocabSize'       => $this->vocabSize,
            'fitted'          => $this->fitted,
            'fittedColumnIdx' => $this->fittedColumnIdx,
            'vocabB64'        => $vocabB64,
        ];
    }

    public function setPhpState(array $state): void
    {
        $this->vocabSize       = (int)  ($state['vocabSize']       ?? 0);
        $this->fitted          = (bool) ($state['fitted']          ?? false);
        $this->fittedColumnIdx = isset($state['fittedColumnIdx']) ? (int) $state['fittedColumnIdx'] : null;

        if (!empty($state['vocabB64'])) {
            $tmp = tempnam(\sys_get_temp_dir(), 'pml_vocab_');
            \file_put_contents($tmp, \base64_decode($state['vocabB64']));
            $this->vocabPtr = TensorEngine::get()->vocab_load($tmp);
            \unlink($tmp);
        }
    }

    // __sleep/__wakeup kept for legacy compatibility (PHP serialize() path).
    public function __sleep(): array
    {
        return ['maxFeatures', 'textColumn', 'vocabSize', 'fitted', 'fittedColumnIdx'];
    }

    public function __wakeup(): void {} // vocabPtr lost on legacy path.

    public function __destruct()
    {
        if ($this->vocabPtr !== null) {
            TensorEngine::get()->vocab_free($this->vocabPtr);
            $this->vocabPtr = null;
        }
        if ($this->vocabFilePath !== null && file_exists($this->vocabFilePath)) {
            unlink($this->vocabFilePath);
            $this->vocabFilePath = null;
        }
    }
}
