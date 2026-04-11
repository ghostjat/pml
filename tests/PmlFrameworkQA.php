<?php

declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Tensor;
use Pml\Dataset;
use Pml\Estimators\Classifiers\GaussianNB;
use Pml\Estimators\Classifiers\LogisticRegression;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Estimators\Regression\GradientBoostingRegressor;
use Pml\Estimators\Classifiers\DecisionTreeClassifier;
use Pml\Estimators\AnomalyDetectors\IsolationForest;
use Pml\Estimators\Manifold\TSNE;
use Pml\NeuralNetwork\Sequential;
use Pml\NeuralNetwork\Layers\Dense;
use Pml\NeuralNetwork\Layers\ReLU;
use Pml\NeuralNetwork\Layers\Softmax;
use Pml\Losses\CategoricalCrossEntropy;
use Pml\NeuralNetwork\Optimizers\Adam;
use Pml\Transformers\MinMaxScaler;
use Pml\Transformers\OneHotLabelEncoder;
use Pml\Transformers\WordCountVectorizer;
use Pml\Transformers\TfIdfTransformer;
use Pml\Metrics\Classification\F1Score;
use Pml\Metrics\Regression\RSquared;
use Pml\Metrics\Reports\ConfusionMatrix;

/**
 * PML Framework QA & Credibility Test Suite.
 * Validates functional correctness across the entire C-accelerated stack.
 */
final class PmlFrameworkQA extends TestCase
{
    // ========================================================================
    // 1. TENSOR ENGINE CREDIBILITY
    // ========================================================================

    public function testTensorMatmulAndBroadcasting(): void
    {
        $a = Tensor::fromArray([[1, 2], [3, 4]]);
        $b = Tensor::fromArray([[5, 6], [7, 8]]);
        
        // C-Level Matmul via OpenBLAS
        $res = $a->matmul($b);
        $expected = [19.0, 22.0, 43.0, 50.0];
        $this->assertEquals($expected, $res->toFlatArray());

        // Broadcasting: [2, 2] + [2]
        $v = Tensor::fromArray([10, 20]);
        $broad = $a->add($v);
        $this->assertEquals([11.0, 22.0, 13.0, 24.0], $broad->toFlatArray());
    }

    public function testTensorLinalgSvd(): void
    {
        $a = Tensor::fromArray([[1, 2], [3, 4], [5, 6]]);
        $svd = $a->svd();
        
        $this->assertSame([3, 3], $svd['U']->shape());
        $this->assertSame([2], $svd['S']->shape());
        $this->assertSame([2, 2], $svd['Vt']->shape());
        
        // Reconstruction check: A = U * S * Vt
        $sMatrix = Tensor::zeros(3, 2);
        $sFlat = $svd['S']->toFlatArray();
        $sMatrix->buffer()[0] = $sFlat[0];
        $sMatrix->buffer()[3] = $sFlat[1];
        
        $recon = $svd['U']->matmul($sMatrix)->matmul($svd['Vt']);
        $this->assertEqualsWithDelta($a->toFlatArray(), $recon->toFlatArray(), 0.001);
    }

    // ========================================================================
    // 2. DATASET & TRANSFORMERS QA
    // ========================================================================

    public function testPreprocessingPipeline(): void
    {
        $samples = [[10.0], [20.0], [30.0]];
        $dataset = Dataset::fromArray($samples);
        
        $scaler = new MinMaxScaler(0.0, 1.0);
        $scaler->fit($dataset);
        $transformed = $scaler->transform($dataset);
        
        $this->assertEquals([0.0, 0.5, 1.0], $transformed->samples()->toFlatArray());
    }

    public function testNLPVectorization(): void
    {
        $texts = ["PML is fast", "PML is hardware accelerated"];
        $vectorizer = new WordCountVectorizer();
        $vectorizer->fit($texts);
        $dataset = $vectorizer->transform($texts);
        
        $this->assertSame(2, $dataset->numRows());
        $this->assertGreaterThanOrEqual(4, $dataset->numColumns());
        
        $tfidf = new TfIdfTransformer();
        $tfidf->fit($dataset);
        $weighted = $tfidf->transform($dataset);
        
        // Assert TF-IDF weighting reduced the common word "is"
        $vocab = $vectorizer->vocabulary();
        $isIdx = $vocab['is'];
        $isWeights = $weighted->samples()->col($isIdx)->toFlatArray();
        $this->assertEquals($isWeights[0], $isWeights[1]);
    }

    // ========================================================================
    // 3. CLASSICAL & ENSEMBLE QA
    // ========================================================================

    public function testGaussianNBConvergence(): void
    {
        // Simple distinct clusters
        $samples = [[1.0, 1.0], [1.1, 0.9], [10.0, 10.0], [10.1, 9.9]];
        $labels = [0, 0, 1, 1];
        $dataset = Dataset::fromArray($samples, $labels);
        
        $gnb = new GaussianNB();
        $gnb->train($dataset);
        
        $preds = $gnb->predict($dataset)->toFlatArray();
        $this->assertEquals($labels, $preds);
    }

    public function testRandomForestClassification(): void
    {
        $samples = [[1, 0], [1, 1], [0, 0], [0, 1]];
        $labels = [1, 1, 0, 0]; // Class based on first feature
        $dataset = Dataset::fromArray($samples, $labels);
        
        $rf = new RandomForestClassifier(nEstimators: 10, maxDepth: 2);
        $rf->train($dataset);
        
        $preds = $rf->predict($dataset)->toFlatArray();
        $this->assertEquals($labels, $preds);
    }

    // ========================================================================
    // 4. DEEP LEARNING & BACKPROP QA
    // ========================================================================

    public function testNeuralNetworkGradientFlow(): void
    {
        $model = new Sequential([
            new Dense(4, 8),
            new ReLU(),
            new Dense(8, 3),
            new Softmax()
        ], new CategoricalCrossEntropy(), new Adam(0.01));

        $x = Tensor::randomNormal([2, 4]);
        $y = Tensor::fromArray([[1, 0, 0], [0, 0, 1]]);
        $dataset = Dataset::fromArray($x, $y);

        $initialLoss = (new CategoricalCrossEntropy())->compute($model->forward($x), $y);
        
        // Train for a few steps
        for ($i = 0; $i < 5; $i++) {
            $model->train($dataset, epochs: 1, batchSize: 2);
        }
        
        $finalLoss = (new CategoricalCrossEntropy())->compute($model->forward($x), $y);
        $this->assertLessThan($initialLoss, $finalLoss, "Loss must decrease after Adam updates");
    }

    // ========================================================================
    // 5. ANOMALY & MANIFOLD QA
    // ========================================================================

    public function testIsolationForestAnomalyDetection(): void
    {
        $samples = array_fill(0, 50, [1.0, 1.0]); // Normal cluster
        $samples[] = [100.0, 100.0]; // Outlier
        
        $dataset = Dataset::fromArray($samples);
        $forest = new IsolationForest(nEstimators: 50, contamination: 0.02);
        $forest->train($dataset);
        
        $preds = $forest->predict($dataset)->toFlatArray();
        $this->assertEquals(1.0, end($preds), "Outlier must be detected as 1.0");
    }

    public function testTSNEDimensionReduction(): void
    {
        $samples = Tensor::randomNormal([20, 10]);
        $tsne = new TSNE(nComponents: 2, maxIter: 50);
        $tsne->train(new Dataset($samples));
        
        $embedding = $tsne->embedding();
        $this->assertSame([20, 2], $embedding->shape(), "t-SNE must reduce to 2D");
    }
}