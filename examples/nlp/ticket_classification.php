<?php
declare(strict_types=1);
/**
 * SUPPORT TICKET AUTO-ROUTING
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Classify incoming support tickets to the correct team:
 *            Billing, Technical, Account, Shipping, Returns.
 * Pipeline : WordCountVectorizer + TfIdfTransformer → RandomForestClassifier.
 * Business : Manual triage adds 4–8 hours to first-response time.
 *            Auto-routing to 95 %+ accuracy cuts that to seconds,
 *            directly improving NPS and SLA compliance.
 *
 * NLP NOTE : WCV requires ETL mode — text data is loaded via Dataset::load().
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Pipeline;
use Pml\Estimators\Classifiers\RandomForestClassifier;
use Pml\Transformers\WordCountVectorizer;
use Pml\Transformers\TfIdfTransformer;
use Pml\Metrics\Classification\Accuracy;

section('Support Ticket Auto-Routing — TF-IDF + Random Forest');

// ── 1. Ticket corpus ─────────────────────────────────────────────────────────
// 5 categories: 0=Billing, 1=Technical, 2=Account, 3=Shipping, 4=Returns

$corpus = [
    // Billing (0)
    ['I was charged twice for my subscription this month', 0],
    ['My invoice shows an incorrect amount please help', 0],
    ['I cannot find my payment receipt for last month', 0],
    ['Why was I charged before my free trial ended', 0],
    ['Please cancel my subscription and issue a refund', 0],
    ['My credit card was declined but funds are available', 0],
    ['I need an itemised invoice for tax purposes', 0],
    ['I was charged the wrong plan price on renewal', 0],
    // Technical (1)
    ['The app crashes every time I try to open it', 1],
    ['I cannot connect to the server getting timeout error', 1],
    ['The API is returning 500 errors on my requests', 1],
    ['My dashboard is showing blank white page after login', 1],
    ['Integration with Zapier stopped working after update', 1],
    ['Login page keeps redirecting me in an infinite loop', 1],
    ['Export to CSV is failing with an encoding error', 1],
    ['The mobile app is very slow on iOS 17', 1],
    // Account (2)
    ['I need to change the email address on my account', 2],
    ['How do I reset my two factor authentication', 2],
    ['I cannot log in my password keeps saying incorrect', 2],
    ['I need to add a team member to my workspace', 2],
    ['Please help me transfer my account to a different email', 2],
    ['How do I update my billing address and company name', 2],
    ['I need to delete my account and all my data', 2],
    ['How do I upgrade from individual to business plan', 2],
    // Shipping (3)
    ['My order has not arrived after two weeks', 3],
    ['Tracking shows delivered but I never received it', 3],
    ['I need to change my delivery address before shipment', 3],
    ['The estimated delivery date keeps changing in tracking', 3],
    ['Package arrived damaged from the courier', 3],
    ['I never got a shipping confirmation email', 3],
    ['How long does express shipping take to my country', 3],
    ['Can I pick up my order from a local store', 3],
    // Returns (4)
    ['I want to return a product I received last week', 4],
    ['The item I received is not what I ordered', 4],
    ['How do I start a return and get a prepaid label', 4],
    ['I received the wrong size can I exchange it', 4],
    ['My return was delivered back but refund not processed', 4],
    ['I want to return something bought as a gift', 4],
    ['The return window says expired but product was faulty', 4],
    ['How do I return a digital product I never used', 4],
];

// Augment to ~2000 samples
mt_srand(77);
$aug = [];
for ($rep = 0; $rep < 50; $rep++) {
    foreach ($corpus as [$text, $label]) {
        $words = explode(' ', $text);
        if (count($words) > 4 && mt_rand(0, 1)) {
            unset($words[array_rand($words)]);
        }
        $aug[] = [implode(' ', array_values($words)), $label];
    }
}
shuffle($aug);

$split = (int)(count($aug) * 0.8);
$trainData = array_slice($aug, 0, $split);
$testData  = array_slice($aug, $split);

// ── 2. Write CSV files — WCV requires ETL (DataFrame) mode via Dataset::load()
$trainCsv = sys_get_temp_dir() . '/pml_ticket_train.csv';
$testCsv  = sys_get_temp_dir() . '/pml_ticket_test.csv';
$liveCsv  = sys_get_temp_dir() . '/pml_ticket_live.csv';

$writeCsv = function(string $path, array $rows, array $header = ['text', 'label']): void {
    $fp = fopen($path, 'w');
    fputcsv($fp, $header);
    foreach ($rows as $row) fputcsv($fp, $row);
    fclose($fp);
};

$writeCsv($trainCsv, $trainData);
$writeCsv($testCsv,  $testData);

$trainDs = Dataset::load($trainCsv)->withLabelColumn(1);
$testDs  = Dataset::load($testCsv)->withLabelColumn(1);

metric('Training tickets', count($trainData));
metric('Test tickets',     count($testData));

// ── 3. Train ──────────────────────────────────────────────────────────────────
section('Training');
$t0 = microtime(true);

$pipeline = new Pipeline(
    [new WordCountVectorizer(maxFeatures: 300, textColumn: 'text'), new TfIdfTransformer()],
    new RandomForestClassifier(nEstimators: 150, maxDepth: 10)
);
$pipeline->train($trainDs);

metric('Training time', elapsed($t0));

// ── 4. Evaluate ───────────────────────────────────────────────────────────────
section('Evaluation');
$pred   = $pipeline->predict($testDs);
$labels = $testDs->extractLabelTensor();
metric('Accuracy', (new Accuracy())->score($pred, $labels));

// ── 5. Live routing ───────────────────────────────────────────────────────────
section('Live Ticket Routing');

$teams = ['Billing', 'Technical', 'Account', 'Shipping', 'Returns'];

$newTickets = [
    'I have been billed an extra amount I did not authorise',
    'The application is throwing a null pointer exception',
    'I want to add my colleague to my account as an admin',
    'My parcel is showing as in transit for five days now',
    'The shirt I received is a medium but I ordered large',
    'API integration is failing with authentication error 403',
];

$writeCsv($liveCsv, array_map(fn($t) => [$t, 0], $newTickets));
$liveDs = Dataset::load($liveCsv)->withLabelColumn(1);
$preds  = $pipeline->predict($liveDs)->toFlatArray();

printf("\n  %-12s | %s\n", 'Team', 'Ticket');
printf("  %s\n", str_repeat('-', 72));
foreach ($newTickets as $i => $ticket) {
    $teamIdx = (int)round(max(0.0, min(4.0, $preds[$i])));
    printf("  %-12s | %s\n", $teams[$teamIdx] ?? 'Unknown', substr($ticket, 0, 55));
}

echo "\n✓ Done\n";
