<?php
declare(strict_types=1);

namespace Pml\Benchmarks\Model;

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
use Pml\Estimators\Classifiers\LogisticRegression;
use Pml\Estimators\Regression\LinearRegression;

/**
 * Model persistence benchmarks using each model's own save()/load() API.
 *
 * Sequential, RandomForestClassifier, GaussianNB, LogisticRegression, and
 * LinearRegression each implement Persistable with a custom save(string $dir)
 * and static load(string $dir) that handle FFI\CData (Tensor weights) via
 * SafeTensors. PHP's serialize() is never involved.
 *
 * Groups:
 *   model      — all model persistence benchmarks
 *   save       — write path (config.json + safetensors)
 *   load       — read path (deserialise + inject weights)
 *   roundtrip  — save + load + infer end-to-end
 *   nn         — neural network models
 *   estimator  — classical estimator models
 */
#[Bench\BeforeMethods('setUp')]
#[Bench\Groups(['model', 'persistence'])]
final class ModelPersistenceBench
{
    private static Sequential $nn;
    private static RandomForestClassifier $rf;
    private static GaussianNB $gnb;
    private static LogisticRegression $lr;
    private static LinearRegression $linReg;
    private static string $tmpDir;
    private static bool $initialized = false;

    public function setUp(): void
    {
        if (self::$initialized) {
            return;
        }

        self::$tmpDir = \sys_get_temp_dir() . '/pml_model_bench_' . \getmypid();
        @\mkdir(self::$tmpDir, 0700, true);

        $trainNN   = self::makeDataset(500, 20, 3);
        $trainFlat = self::makeDatasetFlat(500, 20, 2);
        $trainReg  = self::makeRegression(500, 10);

        self::$nn = new Sequential([
            new Dense(20, 128),
            new ReLU(),
            new Dense(128, 64),
            new ReLU(),
            new Dense(64, 3),
            new Softmax(),
        ], new CategoricalCrossEntropy(), new Adam(0.001));
        self::$nn->train($trainNN, epochs: 2, batchSize: 64);

        self::$rf = new RandomForestClassifier(nEstimators: 20, maxDepth: 5);
        self::$rf->train($trainFlat);

        self::$gnb = new GaussianNB();
        self::$gnb->train($trainFlat);

        self::$lr = new LogisticRegression(epochs: 20, learningRate: 0.1, batchSize: 64);
        self::$lr->train($trainFlat);

        self::$linReg = new LinearRegression();
        self::$linReg->train($trainReg);

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
        foreach (\glob($dir . '/*') ?: [] as $item) {
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
            $buf[$i * $classes + ($i % $classes)] = 1.0;
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

    private static function makeRegression(int $n, int $d): Dataset
    {
        $X = Tensor::randomNormal([$n, $d]);
        $w = Tensor::randomNormal([$d, 1]);
        $y = $X->matmul($w)->flatten();
        return new Dataset($X, $y);
    }

    // =========================================================================
    // SAVE — model->save($dir) writes config.json + weights
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['model', 'save', 'nn'])]
    public function benchSaveNN(): void
    {
        $dir = self::$tmpDir . '/nn_' . \uniqid();
        self::$nn->save($dir);
        self::rmDir($dir);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['model', 'save', 'estimator'])]
    public function benchSaveRF(): void
    {
        $dir = self::$tmpDir . '/rf_' . \uniqid();
        self::$rf->save($dir);
        self::rmDir($dir);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['model', 'save', 'estimator'])]
    public function benchSaveGNB(): void
    {
        $dir = self::$tmpDir . '/gnb_' . \uniqid();
        self::$gnb->save($dir);
        self::rmDir($dir);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['model', 'save', 'estimator'])]
    public function benchSaveLR(): void
    {
        $dir = self::$tmpDir . '/lr_' . \uniqid();
        self::$lr->save($dir);
        self::rmDir($dir);
    }

    #[Bench\Iterations(3), Bench\Revs(10)]
    #[Bench\Groups(['model', 'save', 'estimator'])]
    public function benchSaveLinearRegression(): void
    {
        $dir = self::$tmpDir . '/linreg_' . \uniqid();
        self::$linReg->save($dir);
        self::rmDir($dir);
    }

    // =========================================================================
    // LOAD — ClassName::load($dir) restores full model state
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['model', 'load', 'nn'])]
    public function benchLoadNN(): void
    {
        $dir = self::$tmpDir . '/nn_load_' . \uniqid();
        self::$nn->save($dir);
        $restored = Sequential::load($dir);
        self::rmDir($dir);
        unset($restored);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['model', 'load', 'estimator'])]
    public function benchLoadRF(): void
    {
        $dir = self::$tmpDir . '/rf_load_' . \uniqid();
        self::$rf->save($dir);
        $restored = RandomForestClassifier::load($dir);
        self::rmDir($dir);
        unset($restored);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['model', 'load', 'estimator'])]
    public function benchLoadGNB(): void
    {
        $dir = self::$tmpDir . '/gnb_load_' . \uniqid();
        self::$gnb->save($dir);
        $restored = GaussianNB::load($dir);
        self::rmDir($dir);
        unset($restored);
    }

    #[Bench\Iterations(3), Bench\Revs(5)]
    #[Bench\Groups(['model', 'load', 'estimator'])]
    public function benchLoadLinearRegression(): void
    {
        $dir = self::$tmpDir . '/linreg_load_' . \uniqid();
        self::$linReg->save($dir);
        $restored = LinearRegression::load($dir);
        self::rmDir($dir);
        unset($restored);
    }

    // =========================================================================
    // ROUNDTRIP + INFERENCE — restore and run predict to verify usability
    // =========================================================================

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['model', 'roundtrip', 'nn'])]
    public function benchRoundtripNNAndInfer(): void
    {
        $dir = self::$tmpDir . '/nn_rt_' . \uniqid();
        self::$nn->save($dir);
        $restored = Sequential::load($dir);
        $x   = Tensor::randomNormal([16, 20]);
        $out = $restored->forward($x);
        self::rmDir($dir);
        unset($restored, $x, $out);
    }

    #[Bench\Iterations(3), Bench\Revs(3)]
    #[Bench\Groups(['model', 'roundtrip', 'estimator'])]
    public function benchRoundtripRFAndPredict(): void
    {
        $dir = self::$tmpDir . '/rf_rt_' . \uniqid();
        self::$rf->save($dir);
        $restored = RandomForestClassifier::load($dir);
        $test = self::makeDatasetFlat(50, 20, 2);
        $restored->predict($test);
        self::rmDir($dir);
        unset($restored, $test);
    }
}
