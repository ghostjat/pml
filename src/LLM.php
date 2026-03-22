<?php

declare(strict_types=1);
namespace Pml;

class LLM {
    // ... components (embeddings, blocks, final_norm, lm_head) ...

    public function generate(array $input_tokens, int $max_new_tokens, float $temperature = 0.7): array {
        $generated = $input_tokens;

        for ($i = 0; $i < $max_new_tokens; $i++) {
            // Forward Pass
            $logits = $this->forwardPass($generated);
            
            // Pluck the logits for the last token only (Next Token Prediction)
            $last_token_logits = $this->extractLastRow($logits);
            
            // Sample
            $next_token = $this->sampleTopK($last_token_logits, $temperature);
            $generated[] = $next_token;
            
            // Optional: Yield token for streaming output to the user
            echo $this->tokenizer->decode([$next_token]);
            flush();
        }

        return $generated;
    }
}

