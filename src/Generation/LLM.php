<?php
declare(strict_types=1);

namespace Pml\Generation;

use Pml\Tensor;
use Pml\Ops;

class LLM
{
    private array $blocks; // Array of CachedTransformerBlock
    private Tensor $tokenEmbeddings;
    private Tensor $lmHead;
    private int $maxSeqLen = 2048;

    public function generate(array $inputTokens, int $maxNewTokens = 100, float $temperature = 0.7): void
    {
        // Initialize one KV cache per Transformer Block
        $dModel = $this->tokenEmbeddings->shape[1];
        $caches = [];
        foreach ($this->blocks as $block) {
            $caches[] = new KVCache($this->maxSeqLen, $dModel);
        }

        // --- Phase 1: Prefill (Process the Prompt) ---
        // We feed the prompt tokens one by one to populate the KV caches.
        $nextToken = null;
        foreach ($inputTokens as $tokenId) {
            $nextToken = $this->forwardStep($tokenId, $caches);
        }

        // --- Phase 2: Autoregressive Decoding ---
        for ($i = 0; $i < $maxNewTokens; $i++) {
            // Forward pass only the LAST predicted token
            $logits = $this->forwardStep($nextToken, $caches);
            
            // Sample
            $nextToken = Sampler::sample($logits, $temperature);
            
            // Output streaming
            echo "Token ID: " . $nextToken . "\n";
            flush();
            
            if ($nextToken === 2) break; // EOS
        }
    }

    /**
     * Executes a single O(N) step through the network.
     */
    private function forwardStep(int $tokenId, array $caches): Tensor
    {
        // 1. Get embedding for the single token (Shape: [1, d_model])
        $hiddenState = $this->getEmbedding($tokenId);

        // 2. Pass through blocks, updating caches
        foreach ($this->blocks as $index => $block) {
            $hiddenState = $block->forward($hiddenState, $caches[$index]);
        }

        // 3. Project to vocabulary (Logits = hiddenState * lmHead^T)
        // Hidden state is [1, d_model]. lmHead is [vocab_size, d_model].
        // Result is [1, vocab_size].
        $logits = Ops::matmul($hiddenState, $this->lmHead, false, true);

        return $logits; // Return the 1D buffer of vocab scores
    }

    private function getEmbedding(int $tokenId): Tensor 
    {
        $dModel = $this->tokenEmbeddings->shape[1];
        $tokenTensor = new Tensor([1, $dModel]);
        
        $sourcePtr = \FFI::cast("float*", \FFI::addr($this->tokenEmbeddings->buffer[$tokenId * $dModel]));
        $ffi = BlasEngine::get()->ffi;
        $ffi->cblas_scopy($dModel, $sourcePtr, 1, $tokenTensor->buffer, 1);
        
        return $tokenTensor;
    }
}