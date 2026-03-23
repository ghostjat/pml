<?php

declare(strict_types=1);

/**
 * ════════════════════════════════════════════════════════════════════════════
 *  examples/titanic.php — Titanic Survival Prediction
 *  A flagship educational walkthrough of the Pml\Classic ML framework.
 * ════════════════════════════════════════════════════════════════════════════
 *
 * "Women and children first" — the famous order aboard the Titanic on the
 * night of 15 April 1912.  1,517 of the 2,224 people aboard perished.
 * Our goal: teach a machine to predict *who* survived using only the
 * passenger manifest — exactly as Kaggle's introductory competition asks.
 *
 * ── What you will learn ──────────────────────────────────────────────────────
 *  Phase 1  Ingestion    DataLoader::load_csv() → raw mixed-type PHP array
 *  EDA      Exploration  DataProfiler::info() and describe() — know your data
 *  Phase 2  Engineering  Declarative preprocessing via ColumnTransformer:
 *                          SimpleImputer(median) → Age, Fare
 *                          OneHotEncoder        → Sex, Embarked
 *                          passthrough          → Pclass, SibSp, Parch
 *                          drop                 → Name, Ticket, Cabin, PassengerId
 *  Phase 3  Tensors      ColumnTransformer.fit_transform() → Float32 Tensor
 *                        DataSplit::train_test_split()
 *  Phase 4  Pipeline     StandardScaler → XGBClassifier (→ RandomForest fallback)
 *  Phase 5  Evaluation   Accuracy, Confusion Matrix, Precision/Recall/F1
 *
 * ── Usage ────────────────────────────────────────────────────────────────────
 *   php examples/titanic.php
 *   php examples/titanic.php path/to/train.csv
 *
 * ── Dataset ──────────────────────────────────────────────────────────────────
 *   https://www.kaggle.com/c/titanic  (download train.csv)
 *   Place at: datasets/titanic/train.csv
 *
 *   Expected columns (Kaggle format):
 *   PassengerId, Survived, Pclass, Name, Sex, Age,
 *   SibSp, Parch, Ticket, Fare, Cabin, Embarked
 */

// ── Autoloader ────────────────────────────────────────────────────────────────

foreach ([__DIR__ . '/../vendor/autoload.php', __DIR__ . '/vendor/autoload.php'] as $al) {
    if (file_exists($al)) { require_once $al; break; }
}

use Pml\Tensor;
use Pml\Classic\Datasets\DataLoader;
use Pml\Classic\Exploration\DataProfiler;
use Pml\Classic\Compose\ColumnTransformer;
use Pml\Classic\Impute\SimpleImputer;
use Pml\Classic\Preprocess\{StandardScaler, OneHotEncoder};
use Pml\Classic\Ensemble\{XGBClassifier, XGBoostBridge, RandomForestClassifier};
use Pml\Classic\ModelSelection\DataSplit;
use Pml\Classic\Reporting\ModelReporter;
use function Pml\Classic\Pipeline\make_pipeline;

// ── Helper ────────────────────────────────────────────────────────────────────

function banner(string $title, string $phase): void
{
    $line = str_repeat('═', 70);
    echo "\n{$line}\n  {$phase}: {$title}\n{$line}\n";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 1: Ingestion
//
//  DataLoader::load_csv() streams the CSV via fgetcsv(), casting numeric
//  cells to float and keeping strings as-is.
//
//  Because the Titanic CSV contains string columns (Sex, Embarked, Name,
//  Cabin, Ticket), DataLoader returns 'data' as a 2D PHP array rather than
//  a Tensor.  This is intentional — the ColumnTransformer in Phase 2 is
//  designed exactly to handle this mixed-type raw array.
//
//  'target' comes back as a Tensor([n]) of 0.0 / 1.0 (Survived is all-numeric).
// ─────────────────────────────────────────────────────────────────────────────

banner('CSV Ingestion', 'Phase 1');

$csvPath = $argv[1] ?? __DIR__ . '/../datasets/titanic/train.csv';

if (!file_exists($csvPath)) {
    fwrite(STDERR,
        "ERROR: CSV not found at '{$csvPath}'.\n" .
        "Download train.csv from https://www.kaggle.com/c/titanic\n" .
        "and place it at datasets/titanic/train.csv, or pass the path as an argument:\n" .
        "  php examples/titanic.php /path/to/train.csv\n"
    );
    exit(1);
}

$bunch        = DataLoader::load_csv($csvPath, target_column: 'Survived');
$rawX         = $bunch['data'];           // 2D PHP array — mixed float + string
$featureNames = $bunch['feature_names'];  // ['PassengerId', 'Pclass', 'Name', ...]
$targetTensor = $bunch['target'];         // Tensor([n]) of 0.0/1.0

// Materialise the target into a flat PHP float[] for ColumnTransformer
// and rebuild as a Tensor[n] for the model.
$n    = count($rawX);
$yArr = [];
for ($i = 0; $i < $targetTensor->size; $i++) {
    $yArr[] = (float) $targetTensor->buffer[$i];
}
$y = Tensor::fromArray($yArr, [$n]);

printf("  Loaded %d passengers × %d features\n", $n, count($featureNames));
printf("  Survival rate: %.1f%%\n", array_sum($yArr) / $n * 100);

// ─────────────────────────────────────────────────────────────────────────────
//  EDA: Exploratory Data Analysis
//
//  Before transforming anything, inspect the raw data to understand:
//    - Which columns have missing values (non-null count < total rows)
//    - Which columns are numeric vs categorical (dtype)
//    - The distribution of numeric features (mean, std, percentiles)
//
//  This is the "know your data" step that data scientists do before choosing
//  imputation strategies, encoding methods, and model families.
//
//  DataProfiler::info()     → column types and missing value counts
//  DataProfiler::describe() → summary statistics for numeric columns
// ─────────────────────────────────────────────────────────────────────────────

banner('Exploratory Data Analysis', 'EDA');

echo "\n── Column Overview (info) ────────────────────────────────────────────\n";
DataProfiler::info($rawX, $featureNames);

echo "\n── Numeric Summary (describe) ────────────────────────────────────────\n";
$stats = DataProfiler::describe($rawX, $featureNames);
DataProfiler::print_describe($stats, precision: 2);

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 2: Feature Engineering via ColumnTransformer
//
//  The ColumnTransformer replaces the manual foreach engineering loop with a
//  declarative pipeline of column-level transformations.  Each entry is a
//  3-tuple: ['name', transformer_or_directive, ['col1', 'col2', ...]].
//
//  ── Why these choices? ────────────────────────────────────────────────────
//
//  SimpleImputer(median) on ['Age', 'Fare']:
//    Age has ~177 missing values (20% of passengers).  The global mean age
//    (~29.7) is pulled up by older first-class passengers.  The per-column
//    median (28.0) better represents the typical passenger and is robust to
//    the outlier octogenarian (age=80).  Fare has only 1 missing value — the
//    median (14.45) is used for consistency.
//
//  OneHotEncoder(handle_unknown='ignore') on ['Sex', 'Embarked']:
//    Sex is binary (male/female) → 2 OHE columns.
//    Embarked (port of embarkation) has 3 values (S/C/Q) → 3 OHE columns.
//    2 Embarked values are missing — handle_unknown='ignore' leaves those
//    rows as all-zeros (neutral — no port signal for those passengers).
//    OHE avoids the ordinal assumption of label encoding: there is no
//    natural ordering between ports or sexes.
//
//  'passthrough' on ['Pclass', 'SibSp', 'Parch']:
//    These are already numeric and informative predictors.  Pclass (1/2/3)
//    is ordinal — treating it as a continuous integer is acceptable here.
//    SibSp + Parch encode family structure, a strong survival predictor.
//
//  remainder='drop':
//    PassengerId, Name, Ticket, Cabin are dropped.  Name is too high-cardinality
//    for direct use (though it encodes title, which we could extract).  Cabin
//    has >70% missing.  Ticket and PassengerId carry no predictive signal.
// ─────────────────────────────────────────────────────────────────────────────

banner('Feature Engineering (ColumnTransformer)', 'Phase 2');

$ct = new ColumnTransformer(
    transformers: [
        // ── Numeric imputation ──────────────────────────────────────────────
        //   Age and Fare may be missing (NaN after DataLoader's '' passthrough).
        //   Median is robust to the skewed age distribution on Titanic.
        ['imputer', new SimpleImputer(strategy: 'median'), ['Age', 'Fare']],

        // ── Categorical encoding ────────────────────────────────────────────
        //   String categories → binary indicator columns.
        //   handle_unknown='ignore' silently zeros out the 2 missing Embarked rows.
        ['encoder', new OneHotEncoder(handle_unknown: 'ignore'), ['Sex', 'Embarked']],

        // ── Numeric passthrough ─────────────────────────────────────────────
        //   These columns need no transformation — they're already clean integers.
        ['passnum', 'passthrough', ['Pclass', 'SibSp', 'Parch']],
    ],
    remainder: 'drop',           // drop PassengerId, Name, Ticket, Cabin
    feature_names: $featureNames,
);

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 3: Tensor Bridge
//
//  ColumnTransformer.fit_transform() does three things in one call:
//    1. Learns statistics (medians, OHE vocabularies) from the raw array.
//    2. Applies each transformer to its assigned columns.
//    3. Horizontally concatenates all outputs → one Float32 Tensor.
//
//  Output column layout (left to right):
//    [Age, Fare | Sex_female, Sex_male | Embarked_C, Embarked_Q, Embarked_S | Pclass, SibSp, Parch]
//     (imputed)   (one-hot, 2 cols)     (one-hot, 3 cols)                    (passthrough, 3 cols)
//                                               → 10 features total
//
//  This Float32 Tensor lives in C memory (FFI-allocated) — the same format
//  that all Pml estimators expect.  No further bridging is needed.
// ─────────────────────────────────────────────────────────────────────────────

banner('Tensor Bridge + Train/Test Split', 'Phase 3');

$X = $ct->fit_transform($rawX, $y);

printf("  Transformed shape: [%s]\n", implode(', ', $X->shape));

// 80/20 stratified split — same random_state as the classic test suites
[$Xtrain, $Xtest, $ytrain, $ytest] = DataSplit::train_test_split(
    $X, $y,
    test_size:    0.20,
    random_state: 42,
);

printf("  Train: %d samples  Test: %d samples\n", $Xtrain->shape[0], $Xtest->shape[0]);

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 4: Pipeline — StandardScaler → Classifier
//
//  StandardScaler is applied after the ColumnTransformer for two reasons:
//    1. The passthrough Pclass/SibSp/Parch columns have different scales than
//       the one-hot columns.  Scaling makes all features comparable.
//    2. SVMs and logistic models need scaling; tree models don't care, but it
//       never hurts and future model swaps won't require pipeline changes.
//
//  Classifier selection:
//    XGBClassifier (n_estimators=100, max_depth=5) — the Kaggle Titanic
//    standard-bearer.  Boosted trees handle non-linear interactions (e.g.
//    female first-class > female third-class) and are robust to mild
//    outliers in Fare.  Kaggle public leaderboard top scores typically use
//    XGBoost with careful feature engineering.
//
//    Falls back to RandomForestClassifier(200 trees) if libxgboost.so is
//    not on LD_LIBRARY_PATH.  Random forests achieve similar accuracy on
//    Titanic without the boosting overhead.
// ─────────────────────────────────────────────────────────────────────────────

banner('Pipeline Training', 'Phase 4');

// Probe for XGBoost availability — same pattern as NativeBindingsSuite
try {
    XGBoostBridge::get();
    $classifier     = new XGBClassifier(
        n_estimators:     100,
        max_depth:        5,
        learning_rate:    0.1,
        subsample:        0.8,
        colsample_bytree: 0.8,
        reg_lambda:       1.0,
        random_state:     42,
    );
    $classifierName = 'XGBClassifier(100 trees, depth=5, lr=0.1, subsample=0.8)';
} catch (\Throwable) {
    $classifier     = new RandomForestClassifier(n_estimators: 200, random_state: 42);
    $classifierName = 'RandomForestClassifier(200 trees)  ← XGBoost unavailable';
}

echo "  Classifier: {$classifierName}\n";

$pipe = make_pipeline(
    new StandardScaler(),
    $classifier,
);

$pipe->fit($Xtrain, $ytrain);
echo "  Pipeline fitted.\n";

// ─────────────────────────────────────────────────────────────────────────────
//  Phase 5: Evaluation via ModelReporter
//
//  ModelReporter.generate() does everything in one call:
//    1. Runs $pipe->predict($Xtest) internally.
//    2. Auto-detects the task as 'classification' (all y values are 0/1 integers).
//    3. Computes the full confusion matrix, accuracy, precision, recall,
//       F1, specificity, and majority-class baseline.
//    4. Derives human-readable insights from metric relationships
//       (e.g. FP > FN bias, conservative/aggressive classifier detection,
//       class imbalance trap warning).
//    5. Returns a pretty-printed JSON string ready to save or ship to an API.
//
//  The JSON report is written to titanic_report.json alongside the script.
// ─────────────────────────────────────────────────────────────────────────────

banner('Evaluation', 'Phase 5');

$jsonReport = ModelReporter::generate(
    model:         $pipe,
    X_test:        $Xtest,
    y_test:        $ytest,
    feature_names: $featureNames,
    task:          'auto',
);

$reportPath = __DIR__ . '/titanic_report.json';
file_put_contents($reportPath, $jsonReport);
echo "  Report saved to {$reportPath}\n\n";

// ── Pretty-print key metrics to stdout ────────────────────────────────────
$report = json_decode($jsonReport, true);
$m      = $report['metrics'];
$cm     = $m['confusion_matrix'];
$base   = $m['baseline'];

echo "  Confusion Matrix\n";
echo "  ╔══════════════════════════════════╗\n";
echo "  ║             │   Actual 0  Actual 1║\n";
echo "  ║─────────────┼──────────  ─────────║\n";
printf("  ║ Predicted 0 │   TN=%4d   FN=%4d ║\n", $cm['TN'], $cm['FN']);
printf("  ║ Predicted 1 │   FP=%4d   TP=%4d ║\n", $cm['FP'], $cm['TP']);
echo "  ╚══════════════════════════════════╝\n\n";

printf("  %-16s  %.4f  (%.1f%%)\n", 'Accuracy',    $m['accuracy'],    $m['accuracy']    * 100);
printf("  %-16s  %.4f  (%.1f%%)\n", 'Precision',   $m['precision'],   $m['precision']   * 100);
printf("  %-16s  %.4f  (%.1f%%)\n", 'Recall',      $m['recall'],      $m['recall']      * 100);
printf("  %-16s  %.4f  (%.1f%%)\n", 'Specificity', $m['specificity'], $m['specificity'] * 100);
printf("  %-16s  %.4f  (%.1f%%)\n", 'F1-score',    $m['f1_score'],    $m['f1_score']    * 100);

echo "\n";
printf("  Majority-class baseline : %.4f (%.1f%%)\n", $base['majority_class_accuracy'], $base['majority_class_accuracy'] * 100);
printf("  Model lift over baseline: %+.2f pp\n", $base['lift_pp']);

echo "\n── Automated Insights ────────────────────────────────────────────────\n";
foreach ($report['insights'] as $i => $insight) {
    printf("  %d. %s\n", $i + 1, wordwrap($insight, 70, "\n     ", true));
}
