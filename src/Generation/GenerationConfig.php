<?php
declare(strict_types=1);

namespace Pml\Generation;
// ═══════════════════════════════════════════════════════════════════════════
//  GENERATION CONFIG
// ═══════════════════════════════════════════════════════════════════════════

final class GenerationConfig
{
    public function __construct(
        public readonly int   $maxNewTokens = 256,
        public readonly float $temperature  = 0.7,
        public readonly int   $topK         = 50,
        public readonly float $topP         = 0.9,
        public readonly int   $eosTokenId   = 2,
        public readonly bool  $stream       = true,
    ) {}
}