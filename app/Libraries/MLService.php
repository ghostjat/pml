<?php

namespace App\Libraries;

use Rubix\ML\Classifiers\RandomForest;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Classifiers\SVC;
use Rubix\ML\Classifiers\NaiveBayes;
use Rubix\ML\Classifiers\AdaBoost;
use Rubix\ML\Classifiers\GradientBoost;
use Rubix\ML\Classifiers\LogisticRegression;
use Rubix\ML\Classifiers\DecisionTree;
use Rubix\ML\Regressors\Ridge;
use Rubix\ML\Regressors\Lasso;
use Rubix\ML\Regressors\SVR;
use Rubix\ML\Regressors\KNNRegressor;
use Rubix\ML\Regressors\RegressionTree;
use Rubix\ML\Regressors\GradientBoostRegressor;
use Rubix\ML\Clusterers\KMeans;
use Rubix\ML\Clusterers\DBSCAN;
use Rubix\ML\Clusterers\GaussianMixture;
use Rubix\ML\Pipeline;
use Rubix\ML\Transformers\ZScaleStandardizer;
use Rubix\ML\Transformers\MinMaxNormalizer;
use Rubix\ML\Transformers\RobustStandardizer;
use Rubix\ML\Transformers\OneHotEncoder;
use Rubix\ML\Transformers\MissingDataImputer;
use Rubix\ML\Transformers\VarianceThresholdFilter;
use Rubix\ML\Transformers\LinearDiscriminantAnalysis;
use Rubix\ML\Transformers\PrincipalComponentAnalysis;
use Rubix\ML\Transformers\PolynomialExpander;
use Rubix\ML\Transformers\NumericStringConverter;
use Rubix\ML\Transformers\TruncatedSVD;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Datasets\Unlabeled;
use Rubix\ML\CrossValidation\KFold;
use Rubix\ML\CrossValidation\StratifiedKFold;
use Rubix\ML\CrossValidation\Metrics\Accuracy;
use Rubix\ML\CrossValidation\Metrics\FBeta;
use Rubix\ML\CrossValidation\Metrics\MCC;
use Rubix\ML\CrossValidation\Metrics\RMSE;
use Rubix\ML\CrossValidation\Metrics\R2;
use Rubix\ML\CrossValidation\Metrics\MeanAbsoluteError;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Helpers\Params;
use League\Csv\Reader;

class MLService
{
    // ─────────────────────────────────────────────────────────
    // ALGORITHM REGISTRY
    // ─────────────────────────────────────────────────────────

    public static function algorithmRegistry(): array
    {
        return [
            // Classifiers
            'random_forest' => [
                'name'     => 'Random Forest',
                'class'    => RandomForest::class,
                'type'     => 'classifier',
                'category' => 'Ensemble',
                'description' => 'An ensemble of decision trees trained via bagging. Highly accurate, handles high-dimensional data well.',
                'params'   => [
                    'estimators' => ['type'=>'int','default'=>100,'min'=>10,'max'=>1000,'label'=>'Number of Trees'],
                    'ratio'      => ['type'=>'float','default'=>0.2,'min'=>0.1,'max'=>0.9,'label'=>'Bag Ratio'],
                    'max_depth'  => ['type'=>'int','default'=>10,'min'=>1,'max'=>50,'label'=>'Max Depth'],
                    'min_samples'=> ['type'=>'int','default'=>3,'min'=>1,'max'=>30,'label'=>'Min Leaf Samples'],
                ],
            ],
            'knn' => [
                'name'     => 'K-Nearest Neighbors',
                'class'    => KNearestNeighbors::class,
                'type'     => 'classifier',
                'category' => 'Instance',
                'description' => 'Classifies based on the k closest training samples. Simple, no training phase.',
                'params'   => [
                    'k'          => ['type'=>'int','default'=>5,'min'=>1,'max'=>50,'label'=>'Neighbors (k)'],
                    'weighted'   => ['type'=>'bool','default'=>false,'label'=>'Distance Weighted'],
                ],
            ],
            'svc' => [
                'name'     => 'Support Vector Classifier',
                'class'    => SVC::class,
                'type'     => 'classifier',
                'category' => 'Kernel',
                'description' => 'Finds optimal hyperplane separating classes. Effective in high-dimensional spaces.',
                'params'   => [
                    'c'      => ['type'=>'float','default'=>1.0,'min'=>0.01,'max'=>100,'label'=>'Regularization C'],
                    'kernel' => ['type'=>'select','default'=>'rbf','options'=>['linear','rbf','poly'],'label'=>'Kernel'],
                ],
            ],
            'naive_bayes' => [
                'name'     => 'Gaussian Naive Bayes',
                'class'    => NaiveBayes::class,
                'type'     => 'classifier',
                'category' => 'Probabilistic',
                'description' => 'Probabilistic classifier using Bayes theorem with strong independence assumptions.',
                'params'   => [
                    'priors'  => ['type'=>'null','default'=>null,'label'=>'Class Priors (null=auto)'],
                ],
            ],
            'ada_boost' => [
                'name'     => 'AdaBoost',
                'class'    => AdaBoost::class,
                'type'     => 'classifier',
                'category' => 'Boosting',
                'description' => 'Adaptive boosting combines weak learners into a strong classifier iteratively.',
                'params'   => [
                    'estimators'  => ['type'=>'int','default'=>100,'min'=>10,'max'=>500,'label'=>'Boosting Rounds'],
                    'learning_rate'=> ['type'=>'float','default'=>1.0,'min'=>0.01,'max'=>2.0,'label'=>'Learning Rate'],
                ],
            ],
            'gradient_boost' => [
                'name'     => 'Gradient Boost Classifier',
                'class'    => GradientBoost::class,
                'type'     => 'classifier',
                'category' => 'Boosting',
                'description' => 'Builds trees sequentially correcting prior errors. Often top-performing on tabular data.',
                'params'   => [
                    'estimators'   => ['type'=>'int','default'=>100,'min'=>10,'max'=>500,'label'=>'Trees'],
                    'learning_rate'=> ['type'=>'float','default'=>0.1,'min'=>0.001,'max'=>1.0,'label'=>'Learning Rate'],
                    'max_depth'    => ['type'=>'int','default'=>3,'min'=>1,'max'=>10,'label'=>'Max Depth'],
                    'ratio'        => ['type'=>'float','default'=>0.8,'min'=>0.1,'max'=>1.0,'label'=>'Sample Ratio'],
                ],
            ],
            'logistic_regression' => [
                'name'     => 'Logistic Regression',
                'class'    => LogisticRegression::class,
                'type'     => 'classifier',
                'category' => 'Linear',
                'description' => 'Linear model for classification using logistic function. Fast, interpretable.',
                'params'   => [
                    'batch_size'    => ['type'=>'int','default'=>128,'min'=>10,'max'=>1000,'label'=>'Batch Size'],
                    'optimizer'     => ['type'=>'select','default'=>'adam','options'=>['adam','sgd'],'label'=>'Optimizer'],
                    'l2_penalty'    => ['type'=>'float','default'=>1e-4,'min'=>0,'max'=>1,'label'=>'L2 Penalty'],
                    'epochs'        => ['type'=>'int','default'=>100,'min'=>10,'max'=>1000,'label'=>'Max Epochs'],
                ],
            ],
            'decision_tree' => [
                'name'     => 'Decision Tree',
                'class'    => DecisionTree::class,
                'type'     => 'classifier',
                'category' => 'Tree',
                'description' => 'Single decision tree — interpretable rules, prone to overfitting without tuning.',
                'params'   => [
                    'max_depth'  => ['type'=>'int','default'=>10,'min'=>1,'max'=>50,'label'=>'Max Depth'],
                    'min_samples'=> ['type'=>'int','default'=>3,'min'=>1,'max'=>50,'label'=>'Min Samples'],
                ],
            ],
            // Regressors
            'ridge' => [
                'name'     => 'Ridge Regression',
                'class'    => Ridge::class,
                'type'     => 'regressor',
                'category' => 'Linear',
                'description' => 'L2-regularized linear regression. Handles collinear features.',
                'params'   => [
                    'alpha' => ['type'=>'float','default'=>1.0,'min'=>0,'max'=>100,'label'=>'Alpha (L2)'],
                ],
            ],
            'lasso' => [
                'name'     => 'Lasso Regression',
                'class'    => Lasso::class,
                'type'     => 'regressor',
                'category' => 'Linear',
                'description' => 'L1-regularized regression that performs automatic feature selection.',
                'params'   => [
                    'alpha' => ['type'=>'float','default'=>1.0,'min'=>0,'max'=>100,'label'=>'Alpha (L1)'],
                ],
            ],
            'svr' => [
                'name'     => 'Support Vector Regressor',
                'class'    => SVR::class,
                'type'     => 'regressor',
                'category' => 'Kernel',
                'description' => 'SVM for regression. Effective on non-linear relationships with kernel trick.',
                'params'   => [
                    'c'       => ['type'=>'float','default'=>1.0,'min'=>0.01,'max'=>100,'label'=>'C'],
                    'epsilon' => ['type'=>'float','default'=>0.1,'min'=>0,'max'=>1,'label'=>'Epsilon'],
                ],
            ],
            'knn_regressor' => [
                'name'     => 'KNN Regressor',
                'class'    => KNNRegressor::class,
                'type'     => 'regressor',
                'category' => 'Instance',
                'description' => 'Predicts by averaging k nearest neighbors\' target values.',
                'params'   => [
                    'k' => ['type'=>'int','default'=>5,'min'=>1,'max'=>50,'label'=>'Neighbors (k)'],
                ],
            ],
            'gradient_boost_regressor' => [
                'name'     => 'Gradient Boost Regressor',
                'class'    => GradientBoostRegressor::class,
                'type'     => 'regressor',
                'category' => 'Boosting',
                'description' => 'Powerful gradient boosting for regression tasks.',
                'params'   => [
                    'estimators'   => ['type'=>'int','default'=>100,'min'=>10,'max'=>500,'label'=>'Trees'],
                    'learning_rate'=> ['type'=>'float','default'=>0.1,'min'=>0.001,'max'=>1.0,'label'=>'Learning Rate'],
                    'max_depth'    => ['type'=>'int','default'=>3,'min'=>1,'max'=>10,'label'=>'Max Depth'],
                ],
            ],
            // Clusterers
            'kmeans' => [
                'name'     => 'K-Means',
                'class'    => KMeans::class,
                'type'     => 'clusterer',
                'category' => 'Centroid',
                'description' => 'Partitions data into k clusters by minimizing intra-cluster variance.',
                'params'   => [
                    'k'          => ['type'=>'int','default'=>3,'min'=>2,'max'=>20,'label'=>'Clusters (k)'],
                    'epochs'     => ['type'=>'int','default'=>300,'min'=>10,'max'=>1000,'label'=>'Max Iterations'],
                    'min_change' => ['type'=>'float','default'=>1e-4,'min'=>0,'max'=>1,'label'=>'Min Change'],
                ],
            ],
            'dbscan' => [
                'name'     => 'DBSCAN',
                'class'    => DBSCAN::class,
                'type'     => 'clusterer',
                'category' => 'Density',
                'description' => 'Density-based clustering — discovers arbitrary shapes, no k needed.',
                'params'   => [
                    'radius'  => ['type'=>'float','default'=>0.5,'min'=>0.01,'max'=>10,'label'=>'Epsilon Radius'],
                    'min_samples' => ['type'=>'int','default'=>5,'min'=>2,'max'=>50,'label'=>'Min Samples'],
                ],
            ],
        ];
    }

    // ─────────────────────────────────────────────────────────
    // TRANSFORMER REGISTRY
    // ─────────────────────────────────────────────────────────

    public static function transformerRegistry(): array
    {
        return [
            'z_scale'         => ['name'=>'Z-Scale Standardizer','class'=>ZScaleStandardizer::class,'desc'=>'Transforms to μ=0, σ=1'],
            'min_max'         => ['name'=>'Min-Max Normalizer','class'=>MinMaxNormalizer::class,'desc'=>'Scales to [0,1] range'],
            'robust'          => ['name'=>'Robust Standardizer','class'=>RobustStandardizer::class,'desc'=>'Uses median/IQR, outlier resistant'],
            'one_hot'         => ['name'=>'One Hot Encoder','class'=>OneHotEncoder::class,'desc'=>'Encodes categorical to binary'],
            'imputer'         => ['name'=>'Missing Data Imputer','class'=>MissingDataImputer::class,'desc'=>'Fills missing values with strategy'],
            'variance_filter' => ['name'=>'Variance Threshold Filter','class'=>VarianceThresholdFilter::class,'desc'=>'Removes low-variance features'],
            'pca'             => ['name'=>'PCA','class'=>PrincipalComponentAnalysis::class,'desc'=>'Principal component dimensionality reduction'],
            'lda'             => ['name'=>'LDA','class'=>LinearDiscriminantAnalysis::class,'desc'=>'Supervised linear dimensionality reduction'],
            'polynomial'      => ['name'=>'Polynomial Expander','class'=>PolynomialExpander::class,'desc'=>'Adds polynomial feature interactions'],
            'numeric_convert' => ['name'=>'Numeric String Converter','class'=>NumericStringConverter::class,'desc'=>'Converts numeric strings to floats'],
        ];
    }

    // ─────────────────────────────────────────────────────────
    // DATASET PARSING
    // ─────────────────────────────────────────────────────────

    public function parseCSV(string $filePath): array
    {
        $csv = Reader::createFromPath($filePath, 'r');
        $csv->setHeaderOffset(0);

        $headers = $csv->getHeader();
        $records = iterator_to_array($csv->getRecords(), false);

        // Detect column types
        $types = [];
        foreach ($headers as $col) {
            $types[$col] = $this->detectColumnType($records, $col);
        }

        return [
            'headers' => $headers,
            'types'   => $types,
            'records' => $records,
            'count'   => count($records),
        ];
    }

    private function detectColumnType(array $records, string $col): string
    {
        $sample = array_slice($records, 0, min(100, count($records)));
        $numeric = 0;
        $total = 0;
        foreach ($sample as $row) {
            $val = $row[$col] ?? '';
            if ($val === '' || $val === null) continue;
            $total++;
            if (is_numeric($val)) $numeric++;
        }
        if ($total === 0) return 'unknown';
        $ratio = $numeric / $total;
        if ($ratio >= 0.95) return 'continuous';
        $unique = count(array_unique(array_column($sample, $col)));
        if ($unique <= 20) return 'categorical';
        return 'string';
    }

    // ─────────────────────────────────────────────────────────
    // PROFILE / EDA
    // ─────────────────────────────────────────────────────────

    public function profileDataset(array $records, array $headers, array $types): array
    {
        $profile = [];
        foreach ($headers as $col) {
            $values = array_column($records, $col);
            $nonNull = array_filter($values, fn($v) => $v !== '' && $v !== null);
            $missing = count($values) - count($nonNull);

            $colProfile = [
                'name'      => $col,
                'type'      => $types[$col],
                'count'     => count($values),
                'missing'   => $missing,
                'missing_pct'=> count($values) > 0 ? round($missing / count($values) * 100, 2) : 0,
                'unique'    => count(array_unique($values)),
            ];

            if ($types[$col] === 'continuous') {
                $nums = array_map('floatval', array_filter($nonNull, 'is_numeric'));
                if (count($nums) > 0) {
                    sort($nums);
                    $n = count($nums);
                    $mean = array_sum($nums) / $n;
                    $variance = array_sum(array_map(fn($x) => ($x - $mean) ** 2, $nums)) / $n;
                    $colProfile += [
                        'min'    => round(min($nums), 4),
                        'max'    => round(max($nums), 4),
                        'mean'   => round($mean, 4),
                        'median' => round($n % 2 === 0 ? ($nums[$n/2-1] + $nums[$n/2])/2 : $nums[(int)($n/2)], 4),
                        'std'    => round(sqrt($variance), 4),
                        'q25'    => round($nums[(int)($n * 0.25)], 4),
                        'q75'    => round($nums[(int)($n * 0.75)], 4),
                        'skewness'=> $this->skewness($nums, $mean, sqrt($variance)),
                        'histogram'=> $this->histogram($nums, 10),
                    ];
                }
            } else {
                $freq = array_count_values(array_map('strval', $nonNull));
                arsort($freq);
                $colProfile['value_counts'] = array_slice($freq, 0, 20, true);
            }

            $profile[$col] = $colProfile;
        }
        return $profile;
    }

    private function skewness(array $nums, float $mean, float $std): float
    {
        if ($std == 0 || count($nums) < 3) return 0;
        $n = count($nums);
        $sum = array_sum(array_map(fn($x) => (($x - $mean) / $std) ** 3, $nums));
        return round($sum / $n, 4);
    }

    private function histogram(array $nums, int $bins): array
    {
        if (empty($nums)) return [];
        $min = min($nums); $max = max($nums);
        if ($min == $max) return [['bin' => $min, 'count' => count($nums)]];
        $binWidth = ($max - $min) / $bins;
        $hist = array_fill(0, $bins, 0);
        foreach ($nums as $v) {
            $idx = min($bins - 1, (int)(($v - $min) / $binWidth));
            $hist[$idx]++;
        }
        $result = [];
        for ($i = 0; $i < $bins; $i++) {
            $result[] = ['bin' => round($min + $i * $binWidth, 3), 'count' => $hist[$i]];
        }
        return $result;
    }

    public function correlationMatrix(array $records, array $headers, array $types): array
    {
        $numericCols = array_filter($headers, fn($h) => $types[$h] === 'continuous');
        $numericCols = array_values($numericCols);
        $matrix = [];
        $colData = [];

        foreach ($numericCols as $col) {
            $vals = array_map('floatval', array_column($records, $col));
            $colData[$col] = $vals;
        }

        foreach ($numericCols as $a) {
            foreach ($numericCols as $b) {
                $r = $this->pearson($colData[$a], $colData[$b]);
                $matrix[$a][$b] = round($r, 4);
            }
        }

        return ['columns' => $numericCols, 'matrix' => $matrix];
    }

    private function pearson(array $x, array $y): float
    {
        $n = count($x);
        if ($n < 2) return 0;
        $mx = array_sum($x) / $n;
        $my = array_sum($y) / $n;
        $num = $den_x = $den_y = 0;
        for ($i = 0; $i < $n; $i++) {
            $dx = $x[$i] - $mx; $dy = $y[$i] - $my;
            $num   += $dx * $dy;
            $den_x += $dx * $dx;
            $den_y += $dy * $dy;
        }
        $denom = sqrt($den_x * $den_y);
        return $denom == 0 ? 0 : $num / $denom;
    }

    // ─────────────────────────────────────────────────────────
    // BUILD RUBIXML DATASET
    // ─────────────────────────────────────────────────────────

    public function buildDataset(array $records, array $features, string $targetCol, array $types): Labeled
    {
        $samples = [];
        $labels  = [];

        foreach ($records as $row) {
            $sample = [];
            $skip = false;
            foreach ($features as $feat) {
                $val = $row[$feat] ?? null;
                if ($val === '' || $val === null) {
                    $val = $types[$feat] === 'continuous' ? NAN : '?';
                }
                $sample[] = $types[$feat] === 'continuous' ? (float)$val : (string)$val;
            }
            $label = $row[$targetCol] ?? null;
            if ($label === null || $label === '') continue;
            $samples[] = $sample;
            $labels[]  = (string)$label;
        }

        return new Labeled($samples, $labels);
    }

    public function buildUnlabeled(array $records, array $features, array $types): Unlabeled
    {
        $samples = [];
        foreach ($records as $row) {
            $sample = [];
            foreach ($features as $feat) {
                $val = $row[$feat] ?? null;
                $sample[] = $types[$feat] === 'continuous' ? (float)($val ?? 0) : (string)($val ?? '?');
            }
            $samples[] = $sample;
        }
        return new Unlabeled($samples);
    }

    // ─────────────────────────────────────────────────────────
    // BUILD TRANSFORMERS
    // ─────────────────────────────────────────────────────────

    public function buildTransformers(array $config): array
    {
        $transformers = [];

        // Always convert numeric strings first
        $transformers[] = new NumericStringConverter();

        // Imputer
        if (!empty($config['imputer']) && $config['imputer'] !== 'none') {
            $transformers[] = new MissingDataImputer();
        }

        // One hot encoder for categoricals
        if (!empty($config['one_hot'])) {
            $transformers[] = new OneHotEncoder();
        }

        // Scaler
        switch ($config['scaler'] ?? 'z_scale') {
            case 'min_max':   $transformers[] = new MinMaxNormalizer(); break;
            case 'robust':    $transformers[] = new RobustStandardizer(); break;
            case 'z_scale':
            default:          $transformers[] = new ZScaleStandardizer(); break;
        }

        // Variance filter
        if (!empty($config['variance_threshold'])) {
            $transformers[] = new VarianceThresholdFilter((float)$config['variance_threshold']);
        }

        // Dimensionality reduction
        if (!empty($config['pca_components'])) {
            $transformers[] = new PrincipalComponentAnalysis((int)$config['pca_components']);
        }

        // Polynomial expansion
        if (!empty($config['polynomial_degree']) && (int)$config['polynomial_degree'] > 1) {
            $transformers[] = new PolynomialExpander((int)$config['polynomial_degree']);
        }

        return $transformers;
    }

    // ─────────────────────────────────────────────────────────
    // BUILD ESTIMATOR
    // ─────────────────────────────────────────────────────────

    public function buildEstimator(string $algoKey, array $params): object
    {
        $registry = self::algorithmRegistry();
        if (!isset($registry[$algoKey])) {
            throw new \InvalidArgumentException("Unknown algorithm: {$algoKey}");
        }

        $class = $registry[$algoKey]['class'];

        return match ($algoKey) {
            'random_forest'  => new RandomForest(
                (int)($params['estimators'] ?? 100),
                null,
                (float)($params['ratio'] ?? 0.2),
                (int)($params['max_depth'] ?? 10),
                (int)($params['min_samples'] ?? 3),
            ),
            'knn'            => new KNearestNeighbors(
                (int)($params['k'] ?? 5),
                (bool)($params['weighted'] ?? false),
            ),
            'svc'            => new SVC(
                (float)($params['c'] ?? 1.0),
            ),
            'naive_bayes'    => new NaiveBayes(),
            'ada_boost'      => new AdaBoost(
                null,
                (int)($params['estimators'] ?? 100),
                (float)($params['learning_rate'] ?? 1.0),
            ),
            'gradient_boost' => new GradientBoost(
                (int)($params['estimators'] ?? 100),
                (float)($params['learning_rate'] ?? 0.1),
                null,
                (float)($params['ratio'] ?? 0.8),
            ),
            'logistic_regression' => new LogisticRegression(
                (int)($params['batch_size'] ?? 128),
                null,
                (float)($params['l2_penalty'] ?? 1e-4),
                (int)($params['epochs'] ?? 100),
            ),
            'decision_tree'  => new DecisionTree(
                (int)($params['max_depth'] ?? 10),
                (int)($params['min_samples'] ?? 3),
            ),
            'ridge'          => new Ridge((float)($params['alpha'] ?? 1.0)),
            'lasso'          => new Lasso((float)($params['alpha'] ?? 1.0)),
            'svr'            => new SVR(
                (float)($params['c'] ?? 1.0),
                (float)($params['epsilon'] ?? 0.1),
            ),
            'knn_regressor'  => new KNNRegressor((int)($params['k'] ?? 5)),
            'gradient_boost_regressor' => new GradientBoostRegressor(
                (int)($params['estimators'] ?? 100),
                (float)($params['learning_rate'] ?? 0.1),
            ),
            'kmeans'         => new KMeans(
                (int)($params['k'] ?? 3),
                (int)($params['epochs'] ?? 300),
                (float)($params['min_change'] ?? 1e-4),
            ),
            'dbscan'         => new DBSCAN(
                (float)($params['radius'] ?? 0.5),
                (int)($params['min_samples'] ?? 5),
            ),
            default => new $class(),
        };
    }

    // ─────────────────────────────────────────────────────────
    // TRAIN + EVALUATE
    // ─────────────────────────────────────────────────────────

    public function trainAndEvaluate(
        string $algoKey,
        array  $params,
        array  $engineerConfig,
        Labeled $dataset,
        int    $cvFolds = 5,
        string $taskType = 'classification'
    ): array {
        $start  = microtime(true);
        $memBefore = memory_get_usage();

        $estimator = $this->buildEstimator($algoKey, $params);
        $transformers = $this->buildTransformers($engineerConfig);

        $pipeline = new Pipeline($transformers, $estimator);

        // Split: 80% train, 20% test
        [$training, $testing] = $dataset->stratifiedSplit(0.8);

        // Train
        $pipeline->train($training);

        // Test predictions
        $predictions = $pipeline->predict($testing);
        $actual = $testing->labels();

        // Metrics
        $metrics = $this->computeMetrics($actual, $predictions, $taskType);

        // Cross-validation
        $cvScores = [];
        if ($cvFolds > 1) {
            try {
                $validator = $taskType === 'classification'
                    ? new StratifiedKFold($cvFolds)
                    : new KFold($cvFolds);
                $metric = $taskType === 'classification' ? new Accuracy() : new RMSE();
                $cvScore = $validator->test($pipeline, $dataset, $metric);
                $cvScores = $this->computeCVScores($pipeline, $dataset, $taskType, $cvFolds);
            } catch (\Throwable $e) {
                $cvScores = ['error' => $e->getMessage()];
            }
        }

        // Confusion matrix (classification)
        $confMatrix = null;
        if ($taskType === 'classification') {
            $confMatrix = $this->buildConfusionMatrix($actual, $predictions);
        }

        // Feature importances (if supported)
        $featureImportances = null;
        if (method_exists($estimator, 'featureImportances')) {
            try {
                $featureImportances = $estimator->featureImportances();
            } catch (\Throwable) {}
        }

        $trainTime = round(microtime(true) - $start, 4);
        $memPeak   = memory_get_peak_usage() - $memBefore;

        return [
            'metrics'             => $metrics,
            'cv_scores'           => $cvScores,
            'confusion_matrix'    => $confMatrix,
            'feature_importances' => $featureImportances,
            'train_time'          => $trainTime,
            'memory_peak'         => $memPeak,
            'pipeline'            => $pipeline,
        ];
    }

    // ─────────────────────────────────────────────────────────
    // METRICS COMPUTATION
    // ─────────────────────────────────────────────────────────

    public function computeMetrics(array $actual, array $predictions, string $taskType): array
    {
        if ($taskType === 'classification') {
            return $this->classificationMetrics($actual, $predictions);
        }
        return $this->regressionMetrics($actual, $predictions);
    }

    private function classificationMetrics(array $actual, array $predictions): array
    {
        $n = count($actual);
        if ($n === 0) return [];

        $correct = 0;
        for ($i = 0; $i < $n; $i++) {
            if ((string)$actual[$i] === (string)$predictions[$i]) $correct++;
        }
        $accuracy = $correct / $n;

        $classes = array_unique(array_merge($actual, $predictions));
        sort($classes);

        $precision = $recall = $f1 = [];
        foreach ($classes as $cls) {
            $tp = $fp = $fn = 0;
            for ($i = 0; $i < $n; $i++) {
                $isActual = ((string)$actual[$i] === (string)$cls);
                $isPred   = ((string)$predictions[$i] === (string)$cls);
                if ($isActual && $isPred) $tp++;
                elseif (!$isActual && $isPred) $fp++;
                elseif ($isActual && !$isPred) $fn++;
            }
            $p = ($tp + $fp) > 0 ? $tp / ($tp + $fp) : 0;
            $r = ($tp + $fn) > 0 ? $tp / ($tp + $fn) : 0;
            $f = ($p + $r) > 0 ? 2 * $p * $r / ($p + $r) : 0;
            $precision[$cls] = round($p, 4);
            $recall[$cls]    = round($r, 4);
            $f1[$cls]        = round($f, 4);
        }

        $macroP = round(array_sum($precision) / count($precision), 4);
        $macroR = round(array_sum($recall)    / count($recall), 4);
        $macroF = round(array_sum($f1)        / count($f1), 4);

        // MCC
        $mcc = $this->mcc($actual, $predictions, $classes);

        return [
            'accuracy'         => round($accuracy, 4),
            'precision_macro'  => $macroP,
            'recall_macro'     => $macroR,
            'f1_macro'         => $macroF,
            'mcc'              => $mcc,
            'per_class'        => [
                'precision' => $precision,
                'recall'    => $recall,
                'f1'        => $f1,
            ],
            'classes'          => $classes,
            'support'          => array_count_values(array_map('strval', $actual)),
        ];
    }

    private function mcc(array $actual, array $predictions, array $classes): float
    {
        if (count($classes) === 2) {
            [$neg, $pos] = $classes;
            $tp=$fp=$tn=$fn=0;
            foreach ($actual as $i => $a) {
                $p = $predictions[$i];
                if ($a==$pos && $p==$pos) $tp++;
                elseif ($a==$neg && $p==$pos) $fp++;
                elseif ($a==$neg && $p==$neg) $tn++;
                elseif ($a==$pos && $p==$neg) $fn++;
            }
            $d = sqrt(($tp+$fp)*($tp+$fn)*($tn+$fp)*($tn+$fn));
            return $d > 0 ? round(($tp*$tn - $fp*$fn) / $d, 4) : 0;
        }
        // Multiclass simplified
        $n = count($actual);
        $correct = array_sum(array_map(fn($i) => $actual[$i]===$predictions[$i]?1:0, range(0,$n-1)));
        return round(($correct / $n - 1 / count($classes)) / (1 - 1 / count($classes)), 4);
    }

    private function regressionMetrics(array $actual, array $predictions): array
    {
        $n = count($actual);
        if ($n === 0) return [];
        $actual = array_map('floatval', $actual);
        $predictions = array_map('floatval', $predictions);
        $mean = array_sum($actual) / $n;
        $mse = array_sum(array_map(fn($i) => ($actual[$i] - $predictions[$i]) ** 2, range(0,$n-1))) / $n;
        $mae = array_sum(array_map(fn($i) => abs($actual[$i] - $predictions[$i]), range(0,$n-1))) / $n;
        $ssTot = array_sum(array_map(fn($v) => ($v - $mean) ** 2, $actual));
        $ssRes = array_sum(array_map(fn($i) => ($actual[$i] - $predictions[$i]) ** 2, range(0,$n-1)));
        $r2 = $ssTot > 0 ? 1 - $ssRes / $ssTot : 0;
        return [
            'rmse' => round(sqrt($mse), 4),
            'mae'  => round($mae, 4),
            'r2'   => round($r2, 4),
            'mse'  => round($mse, 4),
        ];
    }

    private function buildConfusionMatrix(array $actual, array $predictions): array
    {
        $classes = array_unique(array_merge($actual, $predictions));
        sort($classes);
        $matrix = [];
        foreach ($classes as $a) {
            foreach ($classes as $p) {
                $matrix[$a][$p] = 0;
            }
        }
        foreach ($actual as $i => $a) {
            $p = $predictions[$i];
            $matrix[(string)$a][(string)$p]++;
        }
        return ['classes' => $classes, 'matrix' => $matrix];
    }

    private function computeCVScores(object $pipeline, Labeled $dataset, string $taskType, int $folds): array
    {
        // Manual k-fold for simplicity (RubixML validator returns scalar)
        $n = $dataset->numSamples();
        $foldSize = (int)($n / $folds);
        $scores = [];

        for ($fold = 0; $fold < $folds; $fold++) {
            $start = $fold * $foldSize;
            $end   = min($start + $foldSize, $n);
            // Build val/train indices
            $valIdx   = range($start, $end - 1);
            $trainIdx = array_merge(range(0, $start - 1), range($end, $n - 1));
            if (count($trainIdx) < 10 || count($valIdx) < 2) continue;

            try {
                $samples = $dataset->samples();
                $labels  = $dataset->labels();

                $trainSamples = array_map(fn($i) => $samples[$i], $trainIdx);
                $trainLabels  = array_map(fn($i) => $labels[$i],  $trainIdx);
                $valSamples   = array_map(fn($i) => $samples[$i], $valIdx);
                $valLabels    = array_map(fn($i) => $labels[$i],  $valIdx);

                $trainSet = new Labeled($trainSamples, $trainLabels);
                $valSet   = new Unlabeled($valSamples);

                // Clone pipeline for this fold
                $preds = $pipeline->predict($valSet);
                $metrics = $this->computeMetrics(array_values($valLabels), $preds, $taskType);
                $scores[] = $taskType === 'classification' ? $metrics['accuracy'] : $metrics['rmse'];
            } catch (\Throwable) {}
        }

        if (empty($scores)) return ['mean' => 0, 'std' => 0, 'scores' => []];
        $mean = array_sum($scores) / count($scores);
        $std  = sqrt(array_sum(array_map(fn($s) => ($s - $mean) ** 2, $scores)) / count($scores));
        return ['mean' => round($mean, 4), 'std' => round($std, 4), 'scores' => array_map(fn($s) => round($s, 4), $scores)];
    }

    // ─────────────────────────────────────────────────────────
    // HYPER-PARAMETER TUNING (Grid Search)
    // ─────────────────────────────────────────────────────────

    public function gridSearch(
        string  $algoKey,
        array   $paramGrid,
        array   $engineerConfig,
        Labeled $dataset,
        string  $taskType = 'classification',
        string  $metric   = 'accuracy',
        int     $cvFolds  = 5
    ): array {
        $combinations = $this->cartesian($paramGrid);
        $results = [];
        $best = ['score' => $taskType === 'classification' ? -INF : INF, 'params' => []];

        foreach ($combinations as $combo) {
            try {
                $r = $this->trainAndEvaluate($algoKey, $combo, $engineerConfig, $dataset, $cvFolds, $taskType);
                $score = match ($metric) {
                    'accuracy' => $r['metrics']['accuracy'] ?? 0,
                    'f1'       => $r['metrics']['f1_macro'] ?? 0,
                    'rmse'     => $r['metrics']['rmse'] ?? 0,
                    default    => $r['metrics']['accuracy'] ?? 0,
                };

                $isBetter = $metric === 'rmse' ? $score < $best['score'] : $score > $best['score'];

                $results[] = ['params' => $combo, 'score' => round($score, 4), 'metrics' => $r['metrics']];
                if ($isBetter) {
                    $best = ['score' => $score, 'params' => $combo, 'metrics' => $r['metrics']];
                }
            } catch (\Throwable $e) {
                $results[] = ['params' => $combo, 'score' => null, 'error' => $e->getMessage()];
            }
        }

        usort($results, fn($a,$b) => $metric === 'rmse'
            ? ($a['score'] ?? INF) <=> ($b['score'] ?? INF)
            : ($b['score'] ?? -INF) <=> ($a['score'] ?? -INF)
        );

        return ['best' => $best, 'trials' => $results];
    }

    private function cartesian(array $grid): array
    {
        $result = [[]];
        foreach ($grid as $key => $values) {
            $tmp = [];
            foreach ($result as $existing) {
                foreach ($values as $val) {
                    $tmp[] = array_merge($existing, [$key => $val]);
                }
            }
            $result = $tmp;
        }
        return $result;
    }

    // ─────────────────────────────────────────────────────────
    // PERSIST MODEL
    // ─────────────────────────────────────────────────────────

    public function saveModel(object $pipeline, string $path): void
    {
        $persistent = new PersistentModel($pipeline, new Filesystem($path));
        $persistent->save();
    }

    public function loadModel(string $path): PersistentModel
    {
        return PersistentModel::load(new Filesystem($path));
    }
}