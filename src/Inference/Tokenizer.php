<?php
declare(strict_types=1);

namespace Pml\Inference;

use Pml\Tensor;
use Pml\Lib\TensorEngine;

/**
 * Byte-level BPE tokenizer wrapping the C tok_* API.
 *
 * Compatible with HuggingFace tokenizer.json (GPT-2, LLaMA, Mistral, Phi, etc.)
 * and legacy vocab.json + merges.txt format.
 *
 * All heavy work runs in C.  The PHP layer is a thin orchestration wrapper.
 *
 * Usage:
 *   $tok = Tokenizer::fromJson('/path/to/tokenizer.json');
 *   $ids  = $tok->encode('Hello world', addBos: true);   // int[]
 *   $text = $tok->decode($ids);                          // string
 *   $batch = $tok->encodeBatch(['Hello', 'World'], maxLen: 128); // Tensor
 */
final class Tokenizer
{
    /** @var \FFI\CData  Tokenizer* */
    private \FFI\CData $ptr;

    private function __construct(\FFI\CData $ptr)
    {
        $this->ptr = $ptr;
    }

    public function __destruct()
    {
        self::ffi()->tok_free($this->ptr);
    }

    // ── Factories ─────────────────────────────────────────────────────────────

    /**
     * Load from a HuggingFace tokenizer.json file.
     * Handles BPE byte-level models: GPT-2, LLaMA-2/3, Mistral, Phi-3, Qwen.
     */
    public static function fromJson(string $path): self
    {
        $ffi = self::ffi();
        $ptr = $ffi->tok_load_json($path);
        self::checkError($ffi);
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('[Tokenizer] tok_load_json returned NULL');
        }
        return new self($ptr);
    }

    /**
     * Load from separate vocab.json + merges.txt (legacy GPT-2 format).
     */
    public static function fromFiles(string $vocabPath, string $mergesPath): self
    {
        $ffi = self::ffi();
        $ptr = $ffi->tok_load($vocabPath, $mergesPath);
        self::checkError($ffi);
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('[Tokenizer] tok_load returned NULL');
        }
        return new self($ptr);
    }

    // ── Encoding ──────────────────────────────────────────────────────────────

    /**
     * Encode UTF-8 text to token ids.
     *
     * @return int[]
     */
    public function encode(string $text, bool $addBos = false): array
    {
        $ffi   = self::ffi();
        $n_out = $ffi->new('int');
        $raw   = $ffi->tok_encode($this->ptr, $text, $addBos, \FFI::addr($n_out));
        self::checkError($ffi);
        if (\FFI::isNull($raw)) {
            throw new \RuntimeException('[Tokenizer] tok_encode returned NULL');
        }
        $n = $n_out->cdata;
        $ids = [];
        for ($i = 0; $i < $n; $i++) {
            $ids[] = $raw[$i];
        }
        $ffi->free($raw);
        return $ids;
    }

    /**
     * Batch-encode an array of strings in parallel (OpenMP).
     * Returns a Tensor [n_texts × maxLen] INT32, padded with pad_id.
     *
     * @param string[] $texts
     * @param int      $maxLen  0 = auto (pad to longest in batch)
     */
    public function encodeBatch(array $texts, bool $addBos = false, int $maxLen = 0): Tensor
    {
        $ffi    = self::ffi();
        $n      = count($texts);
        $cstrs  = $ffi->new("char*[{$n}]");
        $owned  = [];
        foreach ($texts as $i => $t) {
            $cs        = \FFI::new('char[' . (strlen($t) + 1) . ']', false);
            \FFI::memcpy($cs, $t, strlen($t));
            $cs[strlen($t)] = "\0";
            $cstrs[$i] = $cs;
            $owned[]   = $cs;
        }
        $ptr = $ffi->tok_encode_batch(
            $this->ptr,
            $ffi->cast('const char**', $cstrs),
            $n,
            $addBos,
            $maxLen
        );
        unset($owned);
        self::checkError($ffi);
        if (\FFI::isNull($ptr)) {
            throw new \RuntimeException('[Tokenizer] tok_encode_batch returned NULL');
        }
        return Tensor::wrap($ptr);
    }

    // ── Decoding ──────────────────────────────────────────────────────────────

    /**
     * Decode token ids back to UTF-8 text.
     *
     * @param int[] $ids
     */
    public function decode(array $ids, bool $skipSpecial = true): string
    {
        $ffi = self::ffi();
        $n   = count($ids);
        if ($n === 0) return '';
        $buf = $ffi->new("int32_t[{$n}]");
        foreach ($ids as $i => $id) {
            $buf[$i] = $id;
        }
        $raw = $ffi->tok_decode(
            $this->ptr,
            $ffi->cast('const int32_t*', $buf),
            $n,
            $skipSpecial
        );
        self::checkError($ffi);
        if (\FFI::isNull($raw)) return '';
        $text = \FFI::string($raw);
        $ffi->free($raw);
        return $text;
    }

    // ── Single-token accessors ─────────────────────────────────────────────────

    /** Convert token id → string.  Returns null if out of range. */
    public function idToStr(int $id): ?string
    {
        $ffi = self::ffi();
        $ptr = $ffi->tok_id_to_str($this->ptr, $id);
        if ($ptr === null) return null;
        if ($ptr instanceof \FFI\CData) {
            return \FFI::isNull($ptr) ? null : \FFI::string($ptr);
        }
        return (string)$ptr;
    }

    /** Convert token string → id.  Returns -1 if not found. */
    public function strToId(string $str): int
    {
        return self::ffi()->tok_str_to_id($this->ptr, $str);
    }

    public function isSpecial(int $id): bool
    {
        return (bool)self::ffi()->tok_is_special($this->ptr, $id);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    public function vocabSize(): int { return self::ffi()->tok_vocab_size($this->ptr); }
    public function bosId(): int     { return self::ffi()->tok_bos_id($this->ptr); }
    public function eosId(): int     { return self::ffi()->tok_eos_id($this->ptr); }
    public function padId(): int     { return self::ffi()->tok_pad_id($this->ptr); }
    public function unkId(): int     { return self::ffi()->tok_unk_id($this->ptr); }

    /** @internal used by InferenceSession */
    public function cptr(): \FFI\CData { return $this->ptr; }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private static function ffi(): \FFI
    {
        return TensorEngine::get();
    }

    private static function checkError(\FFI $ffi): void
    {
        if ($ffi->tensor_check_error()) {
            $msg = \FFI::string($ffi->tensor_get_last_error());
            $ffi->tensor_clear_error();
            throw new \RuntimeException('[Tokenizer] ' . $msg);
        }
    }
}
