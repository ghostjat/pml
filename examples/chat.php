<?php
declare(strict_types=1);

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Tensor;
use Pml\IO\SafetensorsLoader;
require_once __DIR__ . '/../TinyGPT.php'; 

echo "==========================================\n";
echo "  Pml Career Counsellor (Nano-SLM)        \n";
echo "==========================================\n";

// ── 1. Load the Exact Tokenizer from Training ─────────────────────────────
$tokenizerPath = __DIR__ . '/../tokenizer.json';

if (!file_exists($tokenizerPath)) {
    die("[Error] tokenizer.json not found! Please run train.php first to generate it.\n");
}

$tokenizer = json_decode(file_get_contents($tokenizerPath), true);
$byteToId  = $tokenizer['byteToId'];
$idToByte  = $tokenizer['idToByte'];
$vocabSize = $tokenizer['vocabSize'];

echo "[System] Loaded tokenizer. Vocabulary Size: {$vocabSize}\n";

// ── 2. Initialize Model & Load Weights ────────────────────────────────────
echo "[System] Loading model weights from Safetensors...\n";
$ckptPath = __DIR__ . '/../career-nano.safetensors';

// IMPORTANT: These now perfectly match the train.php architecture
const INF_D_MODEL  = 256; 
const INF_N_LAYERS = 6;   
const INF_N_HEADS  = 4;  
const D_FF       = 4 * INF_D_MODEL; 

$model = new TinyGPT($vocabSize, INF_D_MODEL, INF_N_LAYERS, INF_N_HEADS);

// Disable gradient tracking for pure inference (saves memory & time)
foreach ($model->getParams() as $p) {
    $p->requiresGrad = false;
}

if (!file_exists($ckptPath)) {
    die("[Error] Model checkpoint {$ckptPath} not found. Please train the model first.\n");
}

$loadedTensors = SafetensorsLoader::load($ckptPath, false);
$namedParams = $model->namedParams();
$ffi = \Pml\BlasEngine::get()->ffi;

$loadedCount = 0;
foreach ($namedParams as $name => $param) {
    if (isset($loadedTensors[$name])) {
        $ffi->cblas_scopy($param->size, $loadedTensors[$name]->buffer, 1, $param->buffer, 1);
        $loadedCount++;
    }
}
echo "[System] Restored {$loadedCount}/" . count($namedParams) . " weight tensors.\n";
echo "[System] Neural Network online and ready.\n\n";

// ── 3. The Interactive REPL Loop ──────────────────────────────────────────
while (true) {
    echo "\033[1;32mStudent:\033[0m ";
    $input = trim(fgets(STDIN));
    
    if (strtolower($input) === 'exit' || strtolower($input) === 'quit') break;
    if (empty($input)) continue;

    // CRITICAL FIX: Format prompt EXACTLY as the model saw it in training
    $prompt = "Q: {$input}\nA: ";
    
    // Encode prompt to IDs
    $promptBytes = array_values(unpack('C*', $prompt));
    $inputIds = [];
    $fallbackId = $byteToId[32] ?? 0; // fallback to space if an unknown character is typed
    
    foreach ($promptBytes as $b) {
        $inputIds[] = $byteToId[$b] ?? $fallbackId; 
    }

    echo "\033[1;34mCounsellor:\033[0m ";
    
   // ── 4. Autoregressive Generation (With Temperature & Top-K) ─────────────
    $maxNewTokens = 200;
    $temperature  = 0.7; // Lower = more focused, Higher = more creative
    $topK         = 10;  // Only consider the top 10 most likely next characters
    
    $outputString = "";

    for ($i = 0; $i < $maxNewTokens; $i++) {
        $logits = $model->forward($inputIds); 
        
        $seqLen = count($inputIds);
        $lastTokenOffset = ($seqLen - 1) * $vocabSize;
        
        // Extract logits for the last token
        $stepLogits = [];
        for ($v = 0; $v < $vocabSize; $v++) {
            $stepLogits[$v] = (float)$logits->buffer[$lastTokenOffset + $v];
        }

        // Apply Temperature
        foreach ($stepLogits as $v => $logit) {
            $stepLogits[$v] = $logit / $temperature;
        }

        // Top-K Filtering
        arsort($stepLogits); // Sort descending while maintaining index association
        $stepLogits = array_slice($stepLogits, 0, $topK, true);

        // Softmax
        $maxLogit = max($stepLogits);
        $sumExp = 0.0;
        $probs = [];
        foreach ($stepLogits as $v => $logit) {
            $exp = exp($logit - $maxLogit);
            $probs[$v] = $exp;
            $sumExp += $exp;
        }
        
        // Normalize to probabilities
        foreach ($probs as $v => $exp) {
            $probs[$v] = $exp / $sumExp;
        }

        // Sample from the probability distribution
        $rand = mt_rand() / mt_getrandmax();
        $cumulative = 0.0;
        $bestId = array_key_first($probs); // fallback
        
        foreach ($probs as $v => $p) {
            $cumulative += $p;
            if ($rand <= $cumulative) {
                $bestId = $v;
                break;
            }
        }
        
        $char = chr($idToByte[$bestId]);
        $outputString .= $char;
        echo $char;
        
        $inputIds[] = $bestId;
        
        // Stop condition: Check if the last 7 characters are <|end|>
        if (str_ends_with($outputString, '<|end|>')) {
            // Erase the <|end|> tag from the terminal output for a clean look
            echo "\x08\x08\x08\x08\x08\x08\x08       \x08\x08\x08\x08\x08\x08\x08";
            break;
        }
    }
    echo "\n\n";
}