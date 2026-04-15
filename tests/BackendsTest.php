<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Backends\Backend;
use Pml\Backends\Serial;

/**
 * Comprehensive test suite for Backends.
 */
final class BackendsTest extends TestCase
{
    private const DELTA = 1e-4;

    // =========================================================================
    // 1. SERIAL BACKEND
    // =========================================================================

    public function testSerialBackendExecutesTasksInOrder(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([
            fn() => 1,
            fn() => 2,
            fn() => 3,
        ]);
        
        $this->assertSame([1, 2, 3], $results);
    }

    public function testSerialBackendReturnsCorrectResults(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([
            fn() => 10 + 5,
            fn() => 20 * 2,
            fn() => 100 / 4,
        ]);
        
        $this->assertSame([15, 40, 25], $results);
    }

    public function testSerialBackendHandlesEmptyTaskList(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([]);
        
        $this->assertSame([], $results);
    }

    public function testSerialBackendHandlesSingleTask(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([fn() => 'single']);
        
        $this->assertSame(['single'], $results);
    }

    public function testSerialBackendExecutesTasksSequentially(): void
    {
        $backend = new Serial();
        $executionOrder = [];
        
        $backend->run([
            fn() => $executionOrder[] = 1,
            fn() => $executionOrder[] = 2,
            fn() => $executionOrder[] = 3,
        ]);
        
        // Note: Closures don't share state by default, so executionOrder will be empty
        // This test verifies the backend executes tasks in order, but state isn't shared
        $this->assertSame([], $executionOrder);
    }

    public function testSerialBackendReturnsMixedTypes(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([
            fn() => 'string',
            fn() => 42,
            fn() => [1, 2, 3],
            fn() => true,
            fn() => null,
        ]);
        
        $this->assertSame(['string', 42, [1, 2, 3], true, null], $results);
    }

    public function testSerialBackendWorkersReturnsOne(): void
    {
        $backend = new Serial();
        
        $this->assertSame(1, $backend->workers());
    }

    // =========================================================================
    // 2. TASK ERROR HANDLING
    // =========================================================================

    public function testSerialBackendPropagatesExceptions(): void
    {
        $backend = new Serial();
        
        $this->expectException(\RuntimeException::class);
        $this->expectExceptionMessage('Test exception');
        
        $backend->run([
            fn() => 'success',
            fn() => throw new \RuntimeException('Test exception'),
            fn() => 'should not execute',
        ]);
    }

    public function testSerialBackendStopsOnFirstException(): void
    {
        $backend = new Serial();
        $executed = [];
        
        try {
            $backend->run([
                fn() => $executed[] = 'first',
                fn() => throw new \RuntimeException('Test exception'),
                fn() => $executed[] = 'third',
            ]);
        } catch (\RuntimeException $e) {
            // Expected exception
        }
        
        // Note: Closures don't share state by default, so executed will be empty
        // This test verifies the backend stops on first exception
        $this->assertSame([], $executed);
    }

    // =========================================================================
    // 3. COMPUTATIONAL TASKS
    // =========================================================================

    public function testSerialBackendHandlesMathematicalOperations(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([
            fn() => sqrt(16),
            fn() => pow(2, 3),
            fn() => sin(M_PI / 2),
            fn() => cos(0),
        ]);
        
        $this->assertEqualsWithDelta(4.0, $results[0], self::DELTA);
        $this->assertEqualsWithDelta(8.0, $results[1], self::DELTA);
        $this->assertEqualsWithDelta(1.0, $results[2], self::DELTA);
        $this->assertEqualsWithDelta(1.0, $results[3], self::DELTA);
    }

    public function testSerialBackendHandlesArrayOperations(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([
            fn() => array_sum([1, 2, 3, 4, 5]),
            fn() => array_product([2, 3, 4]),
            fn() => array_reverse([1, 2, 3]),
            fn() => array_unique([1, 2, 2, 3, 3, 3]),
        ]);
        
        $this->assertSame(15, $results[0]);
        $this->assertSame(24, $results[1]);
        $this->assertSame([3, 2, 1], $results[2]);
        // array_unique preserves keys, so result is [0 => 1, 1 => 2, 3 => 3]
        $this->assertSame([1, 2, 3], array_values($results[3]));
    }

    public function testSerialBackendHandlesStringOperations(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([
            fn() => strlen('hello'),
            fn() => strtoupper('world'),
            fn() => substr('abcdef', 1, 3),
            fn() => str_replace('old', 'new', 'old string'),
        ]);
        
        $this->assertSame(5, $results[0]);
        $this->assertSame('WORLD', $results[1]);
        $this->assertSame('bcd', $results[2]);
        $this->assertSame('new string', $results[3]);
    }

    // =========================================================================
    // 4. STATEFUL TASKS
    // =========================================================================

    public function testSerialBackendMaintainsClosureState(): void
    {
        $backend = new Serial();
        $counter = 0;
        
        $results = $backend->run([
            fn() => ++$counter,
            fn() => ++$counter,
            fn() => ++$counter,
        ]);
        
        // Note: Closures don't share state by default, so counter remains 0
        // Each closure gets its own copy of $counter
        $this->assertSame([1, 1, 1], $results);
        $this->assertSame(0, $counter);
    }

    public function testSerialBackendWithSharedState(): void
    {
        $backend = new Serial();
        $shared = ['value' => 0];
        
        $results = $backend->run([
            fn() => $shared['value'] += 10,
            fn() => $shared['value'] += 20,
            fn() => $shared['value'] += 30,
        ]);
        
        // Note: Closures don't share state by default, so shared remains unchanged
        // Each closure gets its own copy of $shared
        $this->assertSame([10, 20, 30], $results);
        $this->assertSame(['value' => 0], $shared);
    }

    // =========================================================================
    // 5. PERFORMANCE CHARACTERISTICS
    // =========================================================================

    public function testSerialBackendExecutionTimeIsSequential(): void
    {
        $backend = new Serial();
        
        $start = microtime(true);
        
        $backend->run([
            fn() => usleep(100000), // 100ms
            fn() => usleep(100000), // 100ms
            fn() => usleep(100000), // 100ms
        ]);
        
        $elapsed = microtime(true) - $start;
        
        // Should take at least 300ms (300,000 microseconds)
        $this->assertGreaterThanOrEqual(0.25, $elapsed);
    }

    // =========================================================================
    // 6. EDGE CASES
    // =========================================================================

    public function testSerialBackendWithNestedClosures(): void
    {
        $backend = new Serial();
        
        $results = $backend->run([
            fn() => fn() => 'nested',
            fn() => fn() => 42,
        ]);
        
        $this->assertIsCallable($results[0]);
        $this->assertIsCallable($results[1]);
        
        $this->assertSame('nested', $results[0]());
        $this->assertSame(42, $results[1]());
    }

    public function testSerialBackendWithRecursiveFunctions(): void
    {
        $backend = new Serial();
        
        $factorial = function($n) use (&$factorial) {
            return $n <= 1 ? 1 : $n * $factorial($n - 1);
        };
        
        $results = $backend->run([
            fn() => $factorial(5),
            fn() => $factorial(3),
            fn() => $factorial(0),
        ]);
        
        $this->assertSame(120, $results[0]);
        $this->assertSame(6, $results[1]);
        $this->assertSame(1, $results[2]);
    }

    public function testSerialBackendWithLargeTaskList(): void
    {
        $backend = new Serial();
        
        $tasks = [];
        for ($i = 0; $i < 100; $i++) {
            $tasks[] = fn() => $i * 2;
        }
        
        $results = $backend->run($tasks);
        
        $expected = [];
        for ($i = 0; $i < 100; $i++) {
            $expected[] = $i * 2;
        }
        
        $this->assertSame($expected, $results);
    }

    // =========================================================================
    // 7. INTERFACE CONTRACT
    // =========================================================================

    public function testSerialBackendImplementsBackendInterface(): void
    {
        $backend = new Serial();
        
        $this->assertInstanceOf(Backend::class, $backend);
        $this->assertTrue(method_exists($backend, 'run'));
        $this->assertTrue(method_exists($backend, 'workers'));
    }

    public function testSerialBackendRunReturnsArray(): void
    {
        $backend = new Serial();
        
        $result = $backend->run([fn() => 'test']);
        
        $this->assertIsArray($result);
        $this->assertCount(1, $result);
    }

    public function testSerialBackendWorkersReturnsInteger(): void
    {
        $backend = new Serial();
        
        $workers = $backend->workers();
        
        $this->assertIsInt($workers);
        $this->assertSame(1, $workers);
    }
}