<?php
declare(strict_types=1);

namespace Pml\Generation;

use Pml\Tensor;
use Pml\Ops;

class SimpleTokenizer
{
    private array $vocab;
    private array $inverseVocab;

    public function __construct(string $vocabJsonPath) {
        $this->vocab = json_decode(file_get_contents($vocabJsonPath), true);
        $this->inverseVocab = array_flip($this->vocab);
    }

    public function encode(string $text): array {
        // Highly simplified: a real BPE tokenizer splits by sub-words.
        // This assumes basic word mapping for architectural demonstration.
        $words = explode(' ', $text);
        $tokens = [];
        foreach ($words as $word) {
            $tokens[] = $this->vocab[$word] ?? $this->vocab['<unk>'];
        }
        return $tokens;
    }

    public function decode(int $tokenId): string {
        return $this->inverseVocab[$tokenId] ?? '';
    }
}

class LLM
{
    private array $blocks; // Array of TransformerBlock
    private Tensor $tokenEmbeddings;
    private Tensor $lmHead; // Final projection to vocabulary
    private SimpleTokenizer $tokenizer;

    // ... Constructor would inject loaded weights ...

    public function generate(string $prompt, int $maxNewTokens = 100, float $temperature = 0.7): void
    {
        $inputTokens = $this->tokenizer->encode($prompt);
        
        echo $prompt . " ";

        for ($i = 0; $i < $maxNewTokens; $i++) {
            // 1. Convert sequence of tokens into Embeddings
            // (In a real implementation, you only forward-pass the *new* token 
            // and rely on a KV Cache for past tokens. This is the naive full-pass).
            $seqLen = count($inputTokens);
            $hiddenState = $this->getEmbeddings($inputTokens); // Returns Tensor [$seqLen, $dModel]
            
            // 2. Forward pass through Transformer Blocks
            foreach ($this->blocks as $block) {
                $hiddenState = $block->forward($hiddenState);
            }
            
            // 3. Pluck the last row (the prediction for the *next* token)
            $lastRow = $hiddenState->getRow($seqLen - 1);
            
            // 4. Project to vocabulary logits: Logits = LastRow * lmHead^T
            $logits = Ops::matmul($lastRow, $this->lmHead, false, true);
            
            // 5. Sample the next token
            $nextToken = Sampler::sample($logits, $temperature);
            $inputTokens[] = $nextToken;
            
            // 6. Stream to console
            echo $this->tokenizer->decode($nextToken) . " ";
            flush(); // Force PHP to output immediately
            
            // Stop condition (assuming 2 is EOS token)
            if ($nextToken === 2) break; 
        }
        echo "\n";
    }

    private function getEmbeddings(array $tokens): Tensor {
        // Simplified: pulls rows from the embedding matrix
        $dModel = $this->tokenEmbeddings->shape[1];
        $out = clone $this->tokenEmbeddings; // Dummy allocation logic
        // ... FFI pointer math to pull specific rows based on token IDs ...
        return $out;
    }
}