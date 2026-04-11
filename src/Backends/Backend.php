<?php
declare(strict_types=1);

namespace Pml\Backends;

/**
 * Parallel execution backend interface.
 * Implementations run a list of callables concurrently (or serially).
 */
interface Backend
{
    /**
     * Submit callables for execution and return their results in order.
     *
     * @param  callable[] $tasks
     * @return mixed[]
     */
    public function run(array $tasks): array;

    /**
     * Maximum number of workers available.
     */
    public function workers(): int;
}
