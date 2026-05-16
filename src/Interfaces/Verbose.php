<?php

declare(strict_types=1);

namespace Pml\Interfaces;

use Psr\Log\LoggerAwareInterface;

/**
 * @deprecated Use Psr\Log\LoggerAwareInterface directly (§19).
 *
 * Verbose is a thin re-declaration of the PSR-3 LoggerAwareInterface.
 * It is kept as a deprecated transparent alias so existing code that
 * type-checks for Verbose keeps working.  It will be removed in the
 * next major version.
 *
 * Replacement:
 *   - Interface: implements \Psr\Log\LoggerAwareInterface
 *   - Trait:     use \Psr\Log\LoggerAwareTrait
 */
interface Verbose extends LoggerAwareInterface
{
    // Intentionally empty — all semantics are in LoggerAwareInterface::setLogger().
}
