<?php 

declare(strict_types=1);
namespace Pml;

// ═══════════════════════════════════════════════════════════════════════════
//  MATRIX CLASS
// ═══════════════════════════════════════════════════════════════════════════

class Matrix implements Tensor
{
    /** @var array<int,array<int,float>> Row-major array-of-arrays */
    private array $a;
    private int   $m;  // rows
    private int   $n;  // cols

    private function __construct(array $a, int $m, int $n)
    {
        $this->a = $a; $this->m = $m; $this->n = $n;
    }

    // ── factories ────────────────────────────────────────────────────────

    /** Build from array-of-arrays with validation */
    public static function build(array $a): self
    {
        $m = count($a);
        if ($m === 0) throw new \InvalidArgumentException("Matrix must have at least 1 row");
        $n = count(reset($a));
        foreach ($a as $i => $row) {
            if (count($row) !== $n)
                throw new \InvalidArgumentException("Row $i has ".count($row)." elements, expected $n");
        }
        $out = [];
        foreach ($a as $i => $row) { $r = []; foreach ($row as $v) $r[] = (float)$v; $out[$i] = $r; }
        return new self($out, $m, $n);
    }

    /** Build without validation (trusted input) */
    public static function quick(array $a): self
    {
        $m = count($a); $n = $m > 0 ? count($a[0]) : 0;
        return new self($a, $m, $n);
    }

    public static function zeros(int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) $a[$i] = array_fill(0, $n, 0.0);
        return new self($a, $m, $n);
    }

    public static function ones(int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) $a[$i] = array_fill(0, $n, 1.0);
        return new self($a, $m, $n);
    }

    public static function fill(float $val, int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) $a[$i] = array_fill(0, $n, $val);
        return new self($a, $m, $n);
    }

    public static function eye(int $n): self
    {
        return new self(LinAlg::eye($n), $n, $n);
    }

    public static function identity(int $n): self { return self::eye($n); }

    public static function diagonal(array $elements): self
    {
        $n = count($elements);
        $a = [];
        for ($i = 0; $i < $n; ++$i) {
            $row = array_fill(0, $n, 0.0); $row[$i] = (float)$elements[$i]; $a[$i] = $row;
        }
        return new self($a, $n, $n);
    }

    public static function rand(int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) $row[$j] = mt_rand() / mt_getrandmax();
            $a[$i] = $row;
        }
        return new self($a, $m, $n);
    }

    public static function gaussian(int $m, int $n): self
    {
        $a = [];
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) {
                // Box-Muller
                $u = (mt_rand() + 1) / (mt_getrandmax() + 1);
                $v = (mt_rand() + 1) / (mt_getrandmax() + 1);
                $row[$j] = sqrt(-2*log($u)) * cos(2*M_PI*$v);
            }
            $a[$i] = $row;
        }
        return new self($a, $m, $n);
    }

    public static function uniform(int $m, int $n, float $lo = -1.0, float $hi = 1.0): self
    {
        $a = [];
        $range = $hi - $lo;
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) $row[$j] = $lo + (mt_rand() / mt_getrandmax()) * $range;
            $a[$i] = $row;
        }
        return new self($a, $m, $n);
    }

    public static function poisson(int $m, int $n, float $lambda = 1.0): self
    {
        $a = []; $L = exp(-$lambda);
        for ($i = 0; $i < $m; ++$i) {
            $row = [];
            for ($j = 0; $j < $n; ++$j) {
                $k = 0; $p = 1.0;
                do { ++$k; $p *= mt_rand()/mt_getrandmax(); } while ($p > $L);
                $row[$j] = (float)($k - 1);
            }
            $a[$i] = $row;
        }
        return new self($a, $m, $n);
    }

    // ── Tensor interface ──────────────────────────────────────────────────

    public function shape(): array    { return [$this->m, $this->n]; }
    public function shapeString(): string { return "{$this->m} x {$this->n}"; }
    public function isSquare(): bool  { return $this->m === $this->n; }
    public function size(): int       { return $this->m * $this->n; }
    public function ndim(): int       { return 2; }
    public function dtype(): int      { return Tensor::FLOAT; }
    public function m(): int          { return $this->m; }
    public function n(): int          { return $this->n; }
    public function count(): int      { return $this->m; }  // Countable: rows

    public function asArray(): array  { return $this->a; }
    public function jsonSerialize(): array { return $this->a; }
    public function __toString(): string
    {
        $rows = [];
        foreach ($this->a as $row) $rows[] = '[ '.implode(', ', array_map(fn($v)=>round($v,4), $row)).' ]';
        return implode("\n", $rows);
    }

    public function getIterator(): \ArrayIterator
    {
        return new \ArrayIterator($this->a);
    }

    // ── row/column accessors ──────────────────────────────────────────────

    public function rowAsVector(int $index): Vector
    {
        if ($index < 0 || $index >= $this->m)
            throw new \OutOfBoundsException("Row $index out of bounds");
        return Vector::quick($this->a[$index]);
    }

    public function columnAsVector(int $index): ColumnVector
    {
        if ($index < 0 || $index >= $this->n)
            throw new \OutOfBoundsException("Column $index out of bounds");
        $col = [];
        for ($i = 0; $i < $this->m; ++$i) $col[$i] = $this->a[$i][$index];
        return ColumnVector::quick($col);
    }

    public function diagonal(): Vector
    {
        $k = min($this->m, $this->n); $d = [];
        for ($i = 0; $i < $k; ++$i) $d[$i] = $this->a[$i][$i];
        return Vector::quick($d);
    }

    public function flatten(): Vector
    {
        $flat = [];
        foreach ($this->a as $row) foreach ($row as $v) $flat[] = $v;
        return Vector::quick($flat);
    }

    // ── reductions ────────────────────────────────────────────────────────

    public function sum(): float
    {
        $s = 0.0;
        foreach ($this->a as $row) foreach ($row as $v) $s += $v;
        return $s;
    }

    public function product(): float
    {
        $p = 1.0;
        foreach ($this->a as $row) foreach ($row as $v) $p *= $v;
        return $p;
    }

    public function min(): float
    {
        $m = INF;
        foreach ($this->a as $row) foreach ($row as $v) if ($v < $m) $m = $v;
        return $m;
    }

    public function max(): float
    {
        $m = -INF;
        foreach ($this->a as $row) foreach ($row as $v) if ($v > $m) $m = $v;
        return $m;
    }

    public function mean(): float { return $this->sum() / $this->size(); }

    public function median(): float
    {
        $flat = [];
        foreach ($this->a as $row) foreach ($row as $v) $flat[] = $v;
        sort($flat); $sz = count($flat);
        return $sz % 2 === 0 ? ($flat[$sz/2-1] + $flat[$sz/2]) / 2.0 : $flat[($sz-1)/2];
    }

    public function variance(?Tensor $mean = null): float
    {
        $mu = $mean instanceof self ? $mean->mean() : $this->mean();
        $s  = 0.0; $sz = $this->size();
        foreach ($this->a as $row) foreach ($row as $v) $s += ($v - $mu) ** 2;
        return $s / $sz;
    }

    public function trace(): float
    {
        if (!$this->isSquare()) throw new \RuntimeException("Trace requires square matrix");
        $t = 0.0; for ($i = 0; $i < $this->m; ++$i) $t += $this->a[$i][$i]; return $t;
    }

    public function rowSums(): ColumnVector
    {
        $s = [];
        foreach ($this->a as $i => $row) { $t = 0.0; foreach ($row as $v) $t += $v; $s[$i] = $t; }
        return ColumnVector::quick($s);
    }

    public function columnSums(): Vector
    {
        $s = array_fill(0, $this->n, 0.0);
        foreach ($this->a as $row) foreach ($row as $j => $v) $s[$j] += $v;
        return Vector::quick($s);
    }

    public function rowMeans(): ColumnVector
    {
        return $this->rowSums()->divide($this->n);
    }

    public function columnMeans(): Vector
    {
        return $this->columnSums()->divide($this->m);
    }

    public function rowMaxima(): ColumnVector
    {
        $s = [];
        foreach ($this->a as $i => $row) $s[$i] = max($row);
        return ColumnVector::quick($s);
    }

    public function columnMaxima(): Vector
    {
        $s = array_fill(0, $this->n, -INF);
        foreach ($this->a as $row)
            foreach ($row as $j => $v) if ($v > $s[$j]) $s[$j] = $v;
        return Vector::quick($s);
    }

    public function rowMinima(): ColumnVector
    {
        $s = [];
        foreach ($this->a as $i => $row) $s[$i] = min($row);
        return ColumnVector::quick($s);
    }

    public function columnMinima(): Vector
    {
        $s = array_fill(0, $this->n, INF);
        foreach ($this->a as $row)
            foreach ($row as $j => $v) if ($v < $s[$j]) $s[$j] = $v;
        return Vector::quick($s);
    }

    public function rowNorms(float $p = 2.0): ColumnVector
    {
        $s = [];
        foreach ($this->a as $i => $row) {
            if ($p == 2.0) { $t=0.0; foreach($row as $v) $t+=$v*$v; $s[$i]=sqrt($t); }
            elseif ($p == 1.0) { $t=0.0; foreach($row as $v) $t+=abs($v); $s[$i]=$t; }
            elseif (is_infinite($p)) { $t=0.0; foreach($row as $v) $t=max($t,abs($v)); $s[$i]=$t; }
            else { $t=0.0; foreach($row as $v) $t+=pow(abs($v),$p); $s[$i]=pow($t,1/$p); }
        }
        return ColumnVector::quick($s);
    }

    public function columnNorms(float $p = 2.0): Vector
    {
        $s = array_fill(0, $this->n, 0.0);
        if ($p == 2.0) {
            foreach ($this->a as $row) foreach ($row as $j => $v) $s[$j] += $v*$v;
            for ($j = 0; $j < $this->n; ++$j) $s[$j] = sqrt($s[$j]);
        } elseif ($p == 1.0) {
            foreach ($this->a as $row) foreach ($row as $j => $v) $s[$j] += abs($v);
        } else {
            foreach ($this->a as $row) foreach ($row as $j => $v) $s[$j] += pow(abs($v), $p);
            for ($j = 0; $j < $this->n; ++$j) $s[$j] = pow($s[$j], 1/$p);
        }
        return Vector::quick($s);
    }

    // ── element-wise unary ops ────────────────────────────────────────────

    private function mapElements(callable $fn): self
    {
        $a = [];
        foreach ($this->a as $i => $row) {
            $r = [];
            foreach ($row as $v) $r[] = $fn($v);
            $a[$i] = $r;
        }
        return new self($a, $this->m, $this->n);
    }

    public function round(int $precision = 0): self { return $this->mapElements(fn($v)=>round($v,$precision)); }
    public function floor(): self   { return $this->mapElements('floor'); }
    public function ceil(): self    { return $this->mapElements('ceil'); }
    public function abs(): self     { return $this->mapElements('abs'); }
    public function square(): self  { return $this->mapElements(fn($v)=>$v*$v); }
    public function sqrt(): self    { return $this->mapElements('sqrt'); }
    public function exp(): self     { return $this->mapElements('exp'); }
    public function expm1(): self   { return $this->mapElements('expm1'); }
    public function log(float $base = M_E): self
    {
        return $base === M_E ? $this->mapElements('log') : $this->mapElements(fn($v)=>log($v,$base));
    }
    public function log1p(): self   { return $this->mapElements('log1p'); }
    public function sin(): self     { return $this->mapElements('sin'); }
    public function cos(): self     { return $this->mapElements('cos'); }
    public function tan(): self     { return $this->mapElements('tan'); }
    public function asin(): self    { return $this->mapElements('asin'); }
    public function acos(): self    { return $this->mapElements('acos'); }
    public function atan(): self    { return $this->mapElements('atan'); }
    public function sinh(): self    { return $this->mapElements('sinh'); }
    public function cosh(): self    { return $this->mapElements('cosh'); }
    public function tanh(): self    { return $this->mapElements('tanh'); }
    public function negate(): self  { return $this->mapElements(fn($v)=>-$v); }
    public function reciprocal(): self { return $this->mapElements(fn($v)=>1.0/$v); }
    public function map(callable $fn): self { return $this->mapElements($fn); }

    public function clip(float $min, float $max): self
    { return $this->mapElements(fn($v)=>max($min, min($max, $v))); }
    public function clipLower(float $min): self
    { return $this->mapElements(fn($v)=>max($min, $v)); }
    public function clipUpper(float $max): self
    { return $this->mapElements(fn($v)=>min($max, $v)); }

    public function reduce(callable $fn, int|float $initial = 0): int|float
    {
        $acc = $initial;
        foreach ($this->a as $row) foreach ($row as $v) $acc = $fn($acc, $v);
        return $acc;
    }

    // ── element-wise arithmetic ───────────────────────────────────────────

    public function add(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v+(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a = []; foreach ($this->a as $i=>$row) { $r=[]; foreach($row as $j=>$v) $r[]=$v+$b->a[$i][$j]; $a[$i]=$r; }
            return new self($a, $this->m, $this->n);
        }
        if ($b instanceof Vector) { // broadcast row vector
            if ($b->n() !== $this->n) throw new \RuntimeException("Shape mismatch");
            $bArr = $b->asArray();
            $a = [];
            foreach ($this->a as $i=>$row) { $r=[]; foreach($row as $j=>$v) $r[]=$v+$bArr[$j]; $a[$i]=$r; }
            return new self($a, $this->m, $this->n);
        }
        if ($b instanceof ColumnVector) {
            if ($b->n() !== $this->m) throw new \RuntimeException("Shape mismatch");
            $bArr = $b->asArray();
            $a = [];
            foreach ($this->a as $i=>$row) { $r=[]; foreach($row as $j=>$v) $r[]=$v+$bArr[$i]; $a[$i]=$r; }
            return new self($a, $this->m, $this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function subtract(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v-(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v-$b->a[$i][$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        if ($b instanceof Vector) {
            if ($b->n()!==$this->n) throw new \RuntimeException("Shape mismatch");
            $bArr=$b->asArray(); $a=[];
            foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v-$bArr[$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        if ($b instanceof ColumnVector) {
            if ($b->n()!==$this->m) throw new \RuntimeException("Shape mismatch");
            $bArr=$b->asArray(); $a=[];
            foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v-$bArr[$i];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function multiply(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v*(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v*$b->a[$i][$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        if ($b instanceof Vector) {
            if ($b->n()!==$this->n) throw new \RuntimeException("Shape mismatch");
            $bArr=$b->asArray(); $a=[];
            foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v*$bArr[$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        if ($b instanceof ColumnVector) {
            if ($b->n()!==$this->m) throw new \RuntimeException("Shape mismatch");
            $bArr=$b->asArray(); $a=[];
            foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v*$bArr[$i];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function divide(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v/(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$v/$b->a[$i][$j];$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function pow(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$v**(float)$b);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=pow($v,$b->a[$i][$j]);$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function mod(int|float|Tensor $b): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>fmod($v,(float)$b));
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=fmod($v,$b->a[$i][$j]);$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    // ── comparisons ───────────────────────────────────────────────────────

    private function cmp(int|float|Tensor $b, callable $fn): self
    {
        if (is_numeric($b)) return $this->mapElements(fn($v)=>$fn($v,(float)$b)?1.0:0.0);
        if ($b instanceof self) {
            $this->checkShape($b);
            $a=[]; foreach($this->a as $i=>$row){$r=[];foreach($row as $j=>$v)$r[]=$fn($v,$b->a[$i][$j])?1.0:0.0;$a[$i]=$r;}
            return new self($a,$this->m,$this->n);
        }
        throw new \InvalidArgumentException("Unsupported type");
    }

    public function equal(int|float|Tensor $b): self      { return $this->cmp($b, fn($x,$y)=>$x==$y); }
    public function notEqual(int|float|Tensor $b): self   { return $this->cmp($b, fn($x,$y)=>$x!=$y); }
    public function greater(int|float|Tensor $b): self    { return $this->cmp($b, fn($x,$y)=>$x>$y); }
    public function greaterEqual(int|float|Tensor $b): self { return $this->cmp($b, fn($x,$y)=>$x>=$y); }
    public function less(int|float|Tensor $b): self       { return $this->cmp($b, fn($x,$y)=>$x<$y); }
    public function lessEqual(int|float|Tensor $b): self  { return $this->cmp($b, fn($x,$y)=>$x<=$y); }

    // ── structure ops ─────────────────────────────────────────────────────

    public function transpose(): self
    {
        return new self(LinAlg::transposeRaw($this->a, $this->m, $this->n), $this->n, $this->m);
    }

    public function asType(int $dtype): self
    {
        $fn = $dtype === Tensor::INT ? fn($v)=>(float)(int)$v : fn($v)=>(float)$v;
        return $this->mapElements($fn);
    }

    public function stack(self $b): self
    {
        if ($b->n !== $this->n) throw new \RuntimeException("Column count mismatch");
        return new self(array_merge($this->a, $b->a), $this->m + $b->m, $this->n);
    }

    public function augment(self $b): self
    {
        if ($b->m !== $this->m) throw new \RuntimeException("Row count mismatch");
        $a = [];
        foreach ($this->a as $i => $row) $a[$i] = array_merge($row, $b->a[$i]);
        return new self($a, $this->m, $this->n + $b->n);
    }

    public function repeat(int $m, int $n): self
    {
        $a = [];
        for ($ri = 0; $ri < $this->m * $m; ++$ri) {
            $srcRow = $this->a[$ri % $this->m];
            $row    = [];
            for ($rj = 0; $rj < $n; ++$rj) $row = array_merge($row, $srcRow);
            $a[$ri] = $row;
        }
        return new self($a, $this->m * $m, $this->n * $n);
    }

    public function rowExclude(int $index): self
    {
        $a = $this->a; unset($a[$index]); $a = array_values($a);
        return new self($a, $this->m - 1, $this->n);
    }

    public function columnExclude(int $index): self
    {
        $a = [];
        foreach ($this->a as $i => $row) { unset($row[$index]); $a[$i] = array_values($row); }
        return new self($a, $this->m, $this->n - 1);
    }

    public function softmax(): self
    {
        $a = [];
        foreach ($this->a as $i => $row) {
            $max = max($row);
            $exp = array_map(fn($v)=>exp($v-$max), $row);
            $sum = array_sum($exp);
            $a[$i] = array_map(fn($v)=>$v/$sum, $exp);
        }
        return new self($a, $this->m, $this->n);
    }

    public function symmetric(): bool
    {
        if (!$this->isSquare()) return false;
        for ($i = 0; $i < $this->m; ++$i)
            for ($j = 0; $j < $i; ++$j)
                if (abs($this->a[$i][$j] - $this->a[$j][$i]) > 1e-10) return false;
        return true;
    }

    public function positiveDefinite(): bool
    {
        if (!$this->symmetric()) return false;
        try { $this->cholesky(); return true; } catch (\RuntimeException) { return false; }
    }

    public function positiveHalfDefinite(): bool
    {
        if (!$this->symmetric()) return false;
        try {
            [$w, ] = $this->eig(true);
            foreach ($w->asArray() as $v) if ($v < -1e-10) return false;
            return true;
        } catch (\RuntimeException) { return false; }
    }

    // ── linear algebra ────────────────────────────────────────────────────

    /** Matrix–matrix multiply: returns Matrix (m×p) */
    public function matmul(self $b): self
    {
        if ($this->n !== $b->m) throw new \RuntimeException("Incompatible dimensions for matmul");
        $blas = Blas::getInstance();
        if ($blas->available()) {
            $res = $blas->dgemm($this->a, $this->m, $this->n, $b->a, $b->n);
            return new self($res, $this->m, $b->n);
        }
        return new self(LinAlg::matmul($this->a, $b->a, $this->m, $this->n, $b->n), $this->m, $b->n);
    }

    /** Matrix–vector: A·x → ColumnVector */
    public function dot(Vector $x): ColumnVector
    {
        if ($this->n !== $x->n()) throw new \RuntimeException("Incompatible dimensions for dot");
        $blas = Blas::getInstance();
        if ($blas->available()) {
            $y = $blas->dgemv($this->a, $this->m, $this->n, $x->asArray());
            return ColumnVector::quick($y);
        }
        return ColumnVector::quick(LinAlg::matvec($this->a, $x->asArray(), $this->m, $this->n));
    }

    public function inverse(): self
    {
        if (!$this->isSquare()) throw new \RuntimeException("Matrix must be square");
        $blas = Blas::getInstance();
        if ($blas->available() && $blas->hasLapack()) {
            return new self($blas->inverse($this->a, $this->m), $this->m, $this->m);
        }
        return new self(LinAlg::inverse($this->a, $this->m), $this->m, $this->m);
    }

    public function det(): float
    {
        if (!$this->isSquare()) throw new \RuntimeException("Matrix must be square");
        return LinAlg::det($this->a, $this->m);
    }

    public function rank(): int { return LinAlg::rank($this->a, $this->m, $this->n); }

    public function solve(self $b): self
    {
        if (!$this->isSquare()) throw new \RuntimeException("Coefficient matrix must be square");
        if ($b->m !== $this->m) throw new \RuntimeException("RHS row count mismatch");
        $blas = Blas::getInstance();
        if ($blas->available() && $blas->hasLapack()) {
            $res = $blas->solve($this->a, $b->a, $this->m, $b->n);
            return new self($res, $this->m, $b->n);
        }
        return new self(LinAlg::solve($this->a, $b->a, $this->m, $b->n), $this->m, $b->n);
    }

    /** LU decomposition: returns ['L'=>Matrix, 'U'=>Matrix, 'P'=>Matrix] */
    public function lu(): array
    {
        if (!$this->isSquare()) throw new \RuntimeException("LU requires square matrix");
        $n = $this->m;
        [$LU, $ipiv] = LinAlg::lu($this->a, $n);
        // Separate L and U
        $L = LinAlg::eye($n); $U = [];
        for ($i = 0; $i < $n; ++$i) $U[$i] = array_fill(0, $n, 0.0);
        for ($i = 0; $i < $n; ++$i) {
            for ($j = 0; $j < $n; ++$j) {
                if ($j < $i)       $L[$i][$j] = $LU[$i][$j];
                elseif ($j === $i) { $L[$i][$j] = 1.0; $U[$i][$j] = $LU[$i][$j]; }
                else               $U[$i][$j] = $LU[$i][$j];
            }
        }
        // Build permutation matrix from ipiv
        $pArr = LinAlg::eye($n);
        for ($i = 0; $i < $n; ++$i) if ($ipiv[$i] !== $i) [$pArr[$i], $pArr[$ipiv[$i]]] = [$pArr[$ipiv[$i]], $pArr[$i]];
        return ['L' => new self($L,$n,$n), 'U' => new self($U,$n,$n), 'P' => new self($pArr,$n,$n)];
    }

    public function cholesky(): self
    {
        if (!$this->isSquare()) throw new \RuntimeException("Cholesky requires square matrix");
        $blas = Blas::getInstance();
        if ($blas->available() && $blas->hasLapack()) {
            return new self($blas->dpotrf($this->a, $this->m), $this->m, $this->m);
        }
        return new self(LinAlg::cholesky($this->a, $this->m), $this->m, $this->m);
    }

    /** SVD: returns ['U'=>Matrix, 's'=>Vector, 'VT'=>Matrix] */
    public function svd(): array
    {
        $blas = Blas::getInstance();
        if ($blas->available() && $blas->hasLapack()) {
            [$U, $s, $VT] = $blas->dgesvd($this->a, $this->m, $this->n);
            return ['U' => new self($U, $this->m, $this->m),
                    's' => Vector::quick($s),
                    'VT'=> new self($VT, $this->n, $this->n)];
        }
        [$U, $s, $VT] = LinAlg::svd($this->a, $this->m, $this->n);
        return ['U' => new self($U, $this->m, count($U[0])),
                's' => Vector::quick($s),
                'VT'=> new self($VT, count($VT), $this->n)];
    }

    /**
     * Eigendecomposition: returns ['values'=>Vector, 'vectors'=>Matrix]
     * symmetric=true uses symmetric algorithm (real eigenvalues guaranteed).
     */
    public function eig(bool $symmetric = false): array
    {
        if (!$this->isSquare()) throw new \RuntimeException("Eig requires square matrix");
        $blas = Blas::getInstance();
        if ($symmetric) {
            if ($blas->available() && $blas->hasLapack()) {
                [$w, $v] = $blas->dsyev($this->a, $this->m);
                return ['values' => Vector::quick($w), 'vectors' => new self($v, $this->m, $this->m)];
            }
            [$w, $v] = LinAlg::eigSymmetric($this->a, $this->m);
            return ['values' => Vector::quick($w), 'vectors' => new self($v, $this->m, $this->m)];
        }
        if ($blas->available() && $blas->hasLapack()) {
            [$wr, $wi, $vr] = $blas->dgeev($this->a, $this->m);
            return ['values' => Vector::quick($wr), 'values_imag' => Vector::quick($wi),
                    'vectors' => new self($vr, $this->m, $this->m)];
        }
        // Pure PHP general eig: use symmetric approximation or throw
        throw new \RuntimeException("General eigendecomposition requires LAPACK (set USE_FFI=true and install libopenblas)");
    }

    public function ref(): array
    {
        [$R, $swaps] = LinAlg::ref($this->a, $this->m, $this->n);
        return [new self($R, $this->m, $this->n), $swaps];
    }

    public function rref(): self
    {
        return new self(LinAlg::rref($this->a, $this->m, $this->n), $this->m, $this->n);
    }

    public function pseudoinverse(): self
    {
        ['U'=>$U, 's'=>$s, 'VT'=>$VT] = $this->svd();
        $sArr = $s->asArray(); $tol = 1e-10 * max($this->m,$this->n) * max($sArr);
        $sInv = array_map(fn($v)=>abs($v)>$tol?1.0/$v:0.0, $sArr);
        $SInv = self::diagonal($sInv);
        // A+ = V * S+ * U^T
        return $VT->transpose()->matmul($SInv)->matmul($U->transpose());
    }

    public function convolve(self $kernel, int $stride = 1): self
    {
        $km = $kernel->m; $kn = $kernel->n;
        $om = (int)(($this->m - $km) / $stride) + 1;
        $on = (int)(($this->n - $kn) / $stride) + 1;
        $a  = [];
        for ($i = 0; $i < $om; ++$i) {
            $row = [];
            for ($j = 0; $j < $on; ++$j) {
                $s = 0.0;
                for ($ki = 0; $ki < $km; ++$ki)
                    for ($kj = 0; $kj < $kn; ++$kj)
                        $s += $this->a[$i*$stride+$ki][$j*$stride+$kj] * $kernel->a[$ki][$kj];
                $row[$j] = $s;
            }
            $a[$i] = $row;
        }
        return new self($a, $om, $on);
    }

    // ── helpers ───────────────────────────────────────────────────────────

    private function checkShape(self $b): void
    {
        if ($b->m !== $this->m || $b->n !== $this->n)
            throw new \RuntimeException("Shape mismatch: {$this->m}×{$this->n} vs {$b->m}×{$b->n}");
    }

    #[\Override]
    public function deg2rad(): mixed {
        
    }

    #[\Override]
    public function offsetExists(mixed $offset): bool {
        
    }

    #[\Override]
    public function offsetGet(mixed $offset): mixed {
        
    }

    #[\Override]
    public function offsetSet(mixed $offset, mixed $value): void {
        
    }

    #[\Override]
    public function offsetUnset(mixed $offset): void {
        
    }

    #[\Override]
    public function quantile(float $q): mixed {
        
    }

    #[\Override]
    public function rad2deg(): mixed {
        
    }

    #[\Override]
    public function sign(): mixed {
        
    }
}