<?php
declare(strict_types=1);
/**
 * SENTIMENT ANALYSIS — Product Review Classification
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Classify customer reviews as positive, neutral, negative.
 * Pipeline : WordCountVectorizer + TfIdfTransformer → GBDTClassifier.
 * Business : Aggregated sentiment on 100 k reviews/day drives product
 *            roadmap decisions, vendor scorecards, and NPS dashboards.
 *
 * NLP NOTE : WordCountVectorizer operates on the C-level DataFrame
 *            (ETL mode). Text data is written to a CSV and loaded via
 *            Dataset::load() — the standard PML text-processing path.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Pipeline;
use Pml\Estimators\Classifiers\GBDTClassifier;
use Pml\Transformers\WordCountVectorizer;
use Pml\Transformers\TfIdfTransformer;
use Pml\Metrics\Classification\Accuracy;
use Pml\Metrics\Classification\F1Score;

section('Sentiment Analysis — TF-IDF + GBDT');

// ── 1. Review corpus ─────────────────────────────────────────────────────────
// Label column: 0=negative, 1=neutral, 2=positive

$reviews = [
    // Positive (2)
    ['Absolutely love this product works perfectly and arrived fast', 2],
    ['Excellent quality highly recommend to everyone looking for value', 2],
    ['Amazing customer service resolved my issue in minutes', 2],
    ['Best purchase I have made this year totally worth the price', 2],
    ['Product exceeded my expectations will definitely buy again', 2],
    ['Great build quality and the packaging was superb', 2],
    ['Fast delivery item exactly as described very happy', 2],
    ['Five stars outstanding quality and fantastic support team', 2],
    ['Works flawlessly setup was easy and instructions were clear', 2],
    ['Impressive performance my whole family loves it', 2],
    // Neutral (1)
    ['Product is okay nothing special but does what it says', 1],
    ['Average quality for the price delivery was on time', 1],
    ['It works as expected not great not terrible', 1],
    ['Decent product instructions could have been clearer', 1],
    ['Meets basic requirements nothing to complain about', 1],
    ['Acceptable performance but I expected a bit more', 1],
    ['Standard product delivery was fine no issues so far', 1],
    ['It does the job nothing outstanding about it', 1],
    // Negative (0)
    ['Terrible quality broke after two days of normal use', 0],
    ['Complete waste of money does not work as advertised', 0],
    ['Very disappointed customer service was unhelpful and rude', 0],
    ['Poor construction arrived damaged and return was a nightmare', 0],
    ['Avoid this product cheap materials and bad performance', 0],
    ['Stopped working after one week total garbage', 0],
    ['Not worth the price at all extremely dissatisfied', 0],
    ['Worst purchase ever misleading description and terrible quality', 0],
];

// Augment by repetition with minor word drops
mt_srand(42);
$aug = [];
for ($rep = 0; $rep < 80; $rep++) {
    foreach ($reviews as [$text, $label]) {
        $words = explode(' ', $text);
        if (count($words) > 4 && mt_rand(0, 1)) {
            unset($words[array_rand($words)]);
        }
        $aug[] = [implode(' ', array_values($words)), $label];
    }
}
shuffle($aug);

$split     = (int)(count($aug) * 0.8);
$trainData = array_slice($aug, 0, $split);
$testData  = array_slice($aug, $split);

// ── 2. Write CSV files — WCV requires ETL (DataFrame) mode via Dataset::load()
$trainCsv = sys_get_temp_dir() . '/pml_sentiment_train.csv';
$testCsv  = sys_get_temp_dir() . '/pml_sentiment_test.csv';
$liveCsv  = sys_get_temp_dir() . '/pml_sentiment_live.csv';

$writeCsv = function(string $path, array $rows, array $header = ['text', 'label']): void {
    $fp = fopen($path, 'w');
    fputcsv($fp, $header);
    foreach ($rows as $row) fputcsv($fp, $row);
    fclose($fp);
};

$writeCsv($trainCsv, $trainData);
$writeCsv($testCsv,  $testData);

// Dataset::load() → ETL / DataFrame mode; withLabelColumn(1) marks column 1 as target
$trainDs = Dataset::load($trainCsv)->withLabelColumn(1);
$testDs  = Dataset::load($testCsv)->withLabelColumn(1);

metric('Training reviews', count($trainData));
metric('Test reviews',     count($testData));

// ── 3. Pipeline: WCV → TF-IDF → GBDT ─────────────────────────────────────────
section('Training Pipeline');
$t0 = microtime(true);

$pipeline = new Pipeline(
    [
        new WordCountVectorizer(maxFeatures: 300, textColumn: 'text'),
        new TfIdfTransformer(),
    ],
    new GBDTClassifier(nEstimators: 100, maxDepth: 4, lr: 0.1)
);
$pipeline->train($trainDs);

metric('Training time', elapsed($t0));

// ── 4. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $pipeline->predict($testDs);
$labels = $testDs->extractLabelTensor();

metric('Accuracy', (new Accuracy())->score($pred, $labels));
metric('F1-Score', (new F1Score())->score($pred, $labels));

// ── 5. Live inference ─────────────────────────────────────────────────────────
section('Live Review Scoring');

$newReviews = [
    'This product is absolutely fantastic and I love everything about it',
    'Decent enough for casual use but nothing to write home about',
    'Complete junk stopped working on the very first day I used it',
    'Pretty good value for money I am mostly satisfied with my purchase',
];
$sentiments = ['NEGATIVE', 'NEUTRAL', 'POSITIVE'];

$writeCsv($liveCsv, array_map(fn($t) => [$t, 0], $newReviews));
$liveDs = Dataset::load($liveCsv)->withLabelColumn(1);
$preds  = $pipeline->predict($liveDs)->toFlatArray();

foreach ($newReviews as $i => $text) {
    $label = $sentiments[(int)round(max(0.0, min(2.0, $preds[$i])))] ?? 'UNKNOWN';
    printf("  [%s] \"%s\"\n", str_pad($label, 8), substr($text, 0, 55));
}

echo "\n✓ Done\n";
