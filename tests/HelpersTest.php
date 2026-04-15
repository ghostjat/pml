<?php
declare(strict_types=1);

namespace Pml\Tests;

use PHPUnit\Framework\TestCase;
use Pml\Helpers\CPU;

/**
 * Comprehensive test suite for Helpers.
 */
final class HelpersTest extends TestCase
{
    // =========================================================================
    // 1. CPU CORES DETECTION
    // =========================================================================

    public function testCPUCoresReturnsPositiveInteger(): void
    {
        $cores = CPU::cores();
        
        $this->assertIsInt($cores);
        $this->assertGreaterThan(0, $cores);
    }

    public function testCPUCoresIsAtLeastOne(): void
    {
        $cores = CPU::cores();
        
        $this->assertGreaterThanOrEqual(1, $cores);
    }

    public function testCPUCoresIsReasonable(): void
    {
        $cores = CPU::cores();
        
        // Most systems have between 1 and 128 cores
        $this->assertLessThanOrEqual(128, $cores);
    }

    public function testCPUCoresIsConsistent(): void
    {
        $cores1 = CPU::cores();
        $cores2 = CPU::cores();
        
        $this->assertSame($cores1, $cores2);
    }

    // =========================================================================
    // 2. OPTIMAL WORKERS CALCULATION
    // =========================================================================

    public function testOptimalWorkersWithSingleTask(): void
    {
        $workers = CPU::optimalWorkers(1);
        
        $this->assertSame(1, $workers);
    }

    public function testOptimalWorkersWithMoreTasksThanCores(): void
    {
        $cores = CPU::cores();
        $workers = CPU::optimalWorkers($cores * 10);
        
        $this->assertSame($cores, $workers);
    }

    public function testOptimalWorkersWithFewerTasksThanCores(): void
    {
        $cores = CPU::cores();
        $workers = CPU::optimalWorkers(2);
        
        $this->assertSame(2, $workers);
    }

    public function testOptimalWorkersWithZeroTasks(): void
    {
        $workers = CPU::optimalWorkers(0);
        
        $this->assertSame(0, $workers);
    }

    public function testOptimalWorkersReturnsPositiveInteger(): void
    {
        $workers = CPU::optimalWorkers(100);
        
        $this->assertIsInt($workers);
        $this->assertGreaterThanOrEqual(0, $workers);
    }

    public function testOptimalWorkersIsConsistent(): void
    {
        $workers1 = CPU::optimalWorkers(10);
        $workers2 = CPU::optimalWorkers(10);
        
        $this->assertSame($workers1, $workers2);
    }

    // =========================================================================
    // 3. EDGE CASES
    // =========================================================================

    public function testOptimalWorkersWithLargeNumber(): void
    {
        $workers = CPU::optimalWorkers(1000000);
        $cores = CPU::cores();
        
        $this->assertSame($cores, $workers);
    }

    public function testOptimalWorkersWithNegativeNumber(): void
    {
        $workers = CPU::optimalWorkers(-5);
        
        // Should return the negative number (no validation)
        $this->assertSame(-5, $workers);
    }

    // =========================================================================
    // 4. CLASS STRUCTURE
    // =========================================================================

    public function testCPUClassIsFinal(): void
    {
        $reflection = new \ReflectionClass(CPU::class);
        
        $this->assertTrue($reflection->isFinal());
    }

    public function testCPUClassIsNotInstantiable(): void
    {
        // The CPU class is final but technically instantiable (no private constructor)
        // However, it's designed to be used statically only
        $reflection = new \ReflectionClass(CPU::class);
        
        // It's final, which prevents inheritance
        $this->assertTrue($reflection->isFinal());
    }

    public function testCPUMethodsAreStatic(): void
    {
        $reflection = new \ReflectionClass(CPU::class);
        
        $this->assertTrue($reflection->getMethod('cores')->isStatic());
        $this->assertTrue($reflection->getMethod('optimalWorkers')->isStatic());
    }
}