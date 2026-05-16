<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Serialization;

use PhpBench\Attributes as Bench;
use Pml\Tensor;
use Pml\Dataset;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Estimators\Classifiers\GaussianNB;

/**
 * Model serialization throughput benchmarks.
 *
 * Uses each model's native save(string $dir) / static load(string $dir) API,
 * which writes config.json + weight tensors in SafeTensors format.
 * This is the only correct persistence path — PHP serialize() cannot handle
 * FFI\CData and is not used here.
 *
 * Benchmarks are grouped so you can compare:
 *   - save vs load latency
 *   - small vs large model size impact
 *   - NN vs tree-based vs probabilistic models
 *
 * Groups:
 *   serialization — all benchmarks
 *   save          — write direction
 *   load          — read direction
 *   nn            — Sequential (dense weight tensors via SafeTensors)
 *   estimator     — RF (PHP arrays), GNB (Tensor state)
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['serialization'])]
final class SerializationBench
{
    private static Sequential $smallNN;   // ~10k params
    private static Sequential $largeNN;   // ~600k params
    private static RandomForestClassifier $rf;
    private static GaussianNB $gnb;
    private static string $tmpDir;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        self::$tmpDir = \sys_get_temp_dir() . '/pml_ser_bench_' . \getmypid();
        @\mkdir(self::$tmpDir, 0700, true);

        $trainSmall = self::makeDataset(300, 16, 3);
        $trainLarge = self::makeDataset(500, 64, 5);
        $trainFlat  = self::makeDatasetFlat(500, 20, 2);

        self::$smallNN = new Sequential([
            new Dense(16, 64),
            new ReLU(),
            new Dense(64, 32),
            new ReLU(),
            new Dense(32, 3),
            new Softmax(),
        ], new CategoricalCrossEntropy(), new Adam(0.001));
        self::$smallNN->train($trainSmall, epochs: 1, batchSize: 32);

        self::$largeNN = new Sequential([
            new Dense(64, 512),
            new ReLU(),
            new Dense(512, 512),
            new ReLU(),
            new Dense(512, 256),
            new ReLU(),
            new Dense(256, 5),
            new Softmax(),
        ], new CategoricalCrossEntropy(), new Adam(0.001));
        self::$largeNN->train($trainLarge, epochs: 1, batchSize: 64);

        self::$rf = new RandomForestClassifier(nEstimators: 30, maxDepth: 6);
        self::$rf->train($trainFlat);

        self::$gnb = new GaussianNB();
        self::$gnb->train($trainFlat);

        self::$initialized = true;
    }

    public function __destruct()
    {
        self::rmDir(self::$tmpDir);
    }

    private static function rmDir(string $dir): void
    {
        if (!\is_dir($dir)) {
            return;
        }
        foreach (\glob("$dir/*") ?: [] as $item) {
            \is_dir($item) ? self::rmDir($item) : @\unlink($item);
        }
        @\rmdir($dir);
    }

    private static function makeDataset(int $n, int $d, int $classes): Dataset
    {
        $samples = Tensor::randomNormal([$n, $d]);
        $labels  = Tensor::zeros($n, $classes);
        $buf = $labels->buffer();
        for ($i = 0; $i < $n; $i++) {
            $buf[$i * $classes + $i % $classes] = 1.0;
        }
        return new Dataset($samples, $labels);
    }

    private static function makeDatasetFlat(int $n, int $d, int $classes): Dataset
    {
        $samples = Tensor::randomNormal([$n, $d]);
        $lBuf    = \array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; $i++) {
            $lBuf[$i] = (float)($i % $classes);
        }
        return new Dataset($samples, Tensor::fromArray($lBuf));
    }

    // =========================================================================
    // SAVE — model->save($dir): writes config.json + optional safetensors
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(5)]
    #[Bench\Groups(['serialization', 'save', 'nn'])]
    public function benchSaveSmallNN(): void
    {
        $dir = self::$tmpDir . '/snn_' . \uniqid();
        self::$smallNN->save($dir);
        self::rmDir($dir);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['serialization', 'save', 'nn'])]
    public function benchSaveLargeNN(): void
    {
        $dir = self::$tmpDir . '/lnn_' . \uniqid();
        self::$largeNN->save($dir);
        self::rmDir($dir);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    #[Bench\Groups(['serialization', 'save', 'estimator'])]
    public function benchSaveRF(): void
    {
        $dir = self::$tmpDir . '/rf_' . \uniqid();
        self::$rf->save($dir);
        self::rmDir($dir);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['serialization', 'save', 'estimator'])]
    public function benchSaveGNB(): void
    {
        $dir = self::$tmpDir . '/gnb_' . \uniqid();
        self::$gnb->save($dir);
        self::rmDir($dir);
    }

    // =========================================================================
    // LOAD — ClassName::load($dir): parse JSON + inject SafeTensors weights
    // =========================================================================

    #[Bench\Iterations(5), Bench\Revs(5)]
    #[Bench\Groups(['serialization', 'load', 'nn'])]
    public function benchLoadSmallNN(): void
    {
        $dir = self::$tmpDir . '/snn_l_' . \uniqid();
        self::$smallNN->save($dir);
        $m = Sequential::load($dir);
        self::rmDir($dir);
        unset($m);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['serialization', 'load', 'nn'])]
    public function benchLoadLargeNN(): void
    {
        $dir = self::$tmpDir . '/lnn_l_' . \uniqid();
        self::$largeNN->save($dir);
        $m = Sequential::load($dir);
        self::rmDir($dir);
        unset($m);
    }

    #[Bench\Iterations(5), Bench\Revs(5)]
    #[Bench\Groups(['serialization', 'load', 'estimator'])]
    public function benchLoadRF(): void
    {
        $dir = self::$tmpDir . '/rf_l_' . \uniqid();
        self::$rf->save($dir);
        $m = RandomForestClassifier::load($dir);
        self::rmDir($dir);
        unset($m);
    }

    #[Bench\Iterations(5), Bench\Revs(10)]
    #[Bench\Groups(['serialization', 'load', 'estimator'])]
    public function benchLoadGNB(): void
    {
        $dir = self::$tmpDir . '/gnb_l_' . \uniqid();
        self::$gnb->save($dir);
        $m = GaussianNB::load($dir);
        self::rmDir($dir);
        unset($m);
    }

    // =========================================================================
    // SIZE COMPARISON — save to a fixed path and measure the file sizes emitted
    // (PHPBench mem_peak column reflects PHP memory, not disk bytes)
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['serialization', 'save', 'nn'])]
    public function benchSaveAndVerifySizeSmallNN(): void
    {
        $dir = self::$tmpDir . '/snn_sz_' . \uniqid();
        self::$smallNN->save($dir);
        // Touch the files so PHPBench sees real I/O cost
        $_ = \filesize($dir . '/config.json');
        if (\file_exists($dir . '/model.safetensors')) {
            $_ = \filesize($dir . '/model.safetensors');
        }
        self::rmDir($dir);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['serialization', 'save', 'nn'])]
    public function benchSaveAndVerifySizeLargeNN(): void
    {
        $dir = self::$tmpDir . '/lnn_sz_' . \uniqid();
        self::$largeNN->save($dir);
        $_ = \filesize($dir . '/config.json');
        if (\file_exists($dir . '/model.safetensors')) {
            $_ = \filesize($dir . '/model.safetensors');
        }
        self::rmDir($dir);
    }
}
