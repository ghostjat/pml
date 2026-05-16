<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Inference;

use PhpBench\Attributes as Bench;
use Pml\Inference\Tokenizer;

/**
 * BPE tokenizer performance benchmarks.
 *
 * Measures:
 *   - encode() throughput in tokens/sec and chars/sec
 *   - encodeBatch() for bulk document processing
 *   - decode() latency
 *   - vocab lookup hotpath
 *
 * These benchmarks require a tokenizer.json at the path configured below.
 * If the file is absent, benchmarks are skipped with a warning.
 *
 * To run with a real tokenizer:
 *   export PML_TOKENIZER_PATH=/models/llama3-8b/tokenizer.json
 *   vendor/bin/phpbench run benchmarks/Inference/TokenizerBench.php --report=aggregate
 *
 * A synthetic fallback tokenizer is used when the path is not set —
 * it measures the PHP wrapper and FFI overhead without a real BPE model.
 *
 * Groups:
 *   tokenizer    — all tokenizer benchmarks
 *   encode       — encoding throughput
 *   decode       — decoding latency
 *   batch        — bulk encoding
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['tokenizer', 'inference'])]
final class TokenizerBench
{
    private static ?Tokenizer $tokenizer = null;
    private static bool $available       = false;
    private static bool $initialized     = false;

    // Test strings of varying length and complexity
    private static string $short;    // ~20 tokens
    private static string $medium;   // ~200 tokens
    private static string $long;     // ~2000 tokens
    private static array  $batch50;  // 50 medium strings for batch test
    private static array  $batch500; // 500 short strings

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }
        self::$initialized = true;

        $path = getenv('PML_TOKENIZER_PATH') ?: null;

        if ($path && is_readable($path)) {
            try {
                self::$tokenizer = Tokenizer::fromJson($path);
                self::$available = true;
            } catch (\Throwable $e) {
                echo "[TokenizerBench] Warning: failed to load tokenizer: {$e->getMessage()}\n";
                self::$available = false;
                return;
            }
        } else {
            echo "[TokenizerBench] Note: PML_TOKENIZER_PATH not set or not readable.\n";
            echo "  Set PML_TOKENIZER_PATH=/path/to/tokenizer.json to run real benchmarks.\n";
            echo "  Benchmarks will be skipped.\n";
            self::$available = false;
            return;
        }

        // Build test strings
        self::$short = 'The quick brown fox jumps over the lazy dog.';

        self::$medium = <<<'TEXT'
        Attention mechanisms have become an integral part of compelling sequence modeling and
        transduction models in various tasks, allowing modeling of dependencies without regard
        to their distance in the input or output sequences. In all but a few cases, however,
        such attention mechanisms are used in conjunction with a recurrent network. We propose
        a new simple network architecture, the Transformer, based solely on attention mechanisms,
        dispensing with recurrence and convolutions entirely.
        TEXT;

        // Build a ~2000-token string from repeating technical content
        $paragraph = "Machine learning is a method of data analysis that automates analytical model building. "
            . "It is based on the idea that systems can learn from data, identify patterns and make decisions "
            . "with minimal human intervention. The process of machine learning is similar to that of data mining. "
            . "Both systems search through data to look for patterns. ";
        self::$long = str_repeat($paragraph, 12);

        // Batch inputs
        self::$batch50  = array_fill(0, 50,  self::$medium);
        self::$batch500 = array_fill(0, 500, self::$short);
    }

    // =========================================================================
    // SINGLE ENCODE
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(100)]
    #[Bench\Groups(['tokenizer', 'encode'])]
    public function benchEncodeShort(): void
    {
        if (!self::$available) return;
        $ids = self::$tokenizer->encode(self::$short);
        unset($ids);
    }

    #[Bench\Iterations(5), Bench\Revs(50)]
    #[Bench\Groups(['tokenizer', 'encode'])]
    public function benchEncodeMedium(): void
    {
        if (!self::$available) return;
        $ids = self::$tokenizer->encode(self::$medium);
        unset($ids);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['tokenizer', 'encode'])]
    public function benchEncodeLong(): void
    {
        if (!self::$available) return;
        $ids = self::$tokenizer->encode(self::$long);
        unset($ids);
    }

    // =========================================================================
    // BATCH ENCODE
    //
    // encodeBatch returns a Tensor of shape [N, max_seq_len] (padded).
    // This is the hot path for batch inference preprocessing.
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['tokenizer', 'batch'])]
    public function benchEncodeBatch50xMedium(): void
    {
        if (!self::$available) return;
        $tensor = self::$tokenizer->encodeBatch(self::$batch50);
        unset($tensor);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['tokenizer', 'batch'])]
    public function benchEncodeBatch500xShort(): void
    {
        if (!self::$available) return;
        $tensor = self::$tokenizer->encodeBatch(self::$batch500);
        unset($tensor);
    }

    // =========================================================================
    // DECODE
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(200)]
    #[Bench\Groups(['tokenizer', 'decode'])]
    public function benchDecodeShort(): void
    {
        if (!self::$available) return;
        $ids = self::$tokenizer->encode(self::$short);
        $text = self::$tokenizer->decode($ids->toFlatArray());
        unset($ids, $text);
    }

    #[Bench\Iterations(5), Bench\Revs(50)]
    #[Bench\Groups(['tokenizer', 'decode'])]
    public function benchDecodeMedium(): void
    {
        if (!self::$available) return;
        $ids  = self::$tokenizer->encode(self::$medium);
        $text = self::$tokenizer->decode($ids->toFlatArray());
        unset($ids, $text);
    }

    // =========================================================================
    // VOCAB LOOKUP
    // =========================================================================

    #[Bench\Iterations(10), Bench\Revs(1000)]
    #[Bench\Groups(['tokenizer', 'vocab'])]
    public function benchIdToStr(): void
    {
        if (!self::$available) return;
        // Access a few token IDs — measures hash lookup speed
        self::$tokenizer->idToStr(1);
        self::$tokenizer->idToStr(100);
        self::$tokenizer->idToStr(1000);
    }

    #[Bench\Iterations(10), Bench\Revs(1000)]
    #[Bench\Groups(['tokenizer', 'vocab'])]
    public function benchSpecialTokenQuery(): void
    {
        if (!self::$available) return;
        self::$tokenizer->bosId();
        self::$tokenizer->eosId();
        self::$tokenizer->padId();
        self::$tokenizer->vocabSize();
    }
}
