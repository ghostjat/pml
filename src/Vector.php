<?php 

declare(strict_types=1);
namespace Pml;
// ═══════════════════════════════════════════════════════════════════════════
//  VECTOR CLASS
// ═══════════════════════════════════════════════════════════════════════════

class Vector implements Tensor
{
    protected array $a;
    protected int   $n;

    protected function __construct(array $a, int $n)
    {
        $this->a = $a; $this->n = $n;
    }

    // ── factories ────────────────────────────────────────────────────────

    public static function build(array $a): static
    {
        $flat = []; foreach ($a as $v) $flat[] = (float)$v;
        return new static($flat, count($flat));
    }

    public static function quick(array $a): static
    {
        return new static(array_values($a), count($a));
    }

    public static function zeros(int $n): static
    { return new static(array_fill(0, $n, 0.0), $n); }

    public static function ones(int $n): static
    { return new static(array_fill(0, $n, 1.0), $n); }

    public static function fill(float $val, int $n): static
    { return new static(array_fill(0, $n, $val), $n); }

    public static function rand(int $n): static
    {
        $a = []; for ($i=0;$i<$n;++$i) $a[]= mt_rand()/mt_getrandmax();
        return new static($a, $n);
    }

    public static function gaussian(int $n): static
    {
        $a = [];
        for ($i=0;$i<$n;++$i) {
            $u=(mt_rand()+1)/(mt_getrandmax()+1); $v=(mt_rand()+1)/(mt_getrandmax()+1);
            $a[]=sqrt(-2*log($u))*cos(2*M_PI*$v);
        }
        return new static($a, $n);
    }

    public static function uniform(int $n, float $lo=-1.0, float $hi=1.0): static
    {
        $a=[]; $r=$hi-$lo;
        for($i=0;$i<$n;++$i) $a[]=$lo+(mt_rand()/mt_getrandmax())*$r;
        return new static($a,$n);
    }

    // ── Tensor interface ──────────────────────────────────────────────────

    public function shape(): array   { return [$this->n]; }
    public function shapeString(): string { return (string)$this->n; }
    public function isSquare(): bool { return false; }
    public function size(): int      { return $this->n; }
    public function ndim(): int      { return 1; }
    public function dtype(): int     { return Tensor::FLOAT; }
    public function n(): int         { return $this->n; }
    public function count(): int     { return $this->n; }
    public function asArray(): array { return $this->a; }
    public function jsonSerialize(): array { return $this->a; }
    public function __toString(): string
    { return '[ '.implode(', ', array_map(fn($v)=>round($v,4), $this->a)).' ]'; }

    public function getIterator(): \ArrayIterator
    { return new \ArrayIterator($this->a); }

    // ── conversions ───────────────────────────────────────────────────────

    public function asColumnVector(): ColumnVector { return ColumnVector::quick($this->a); }
    public function asRowMatrix(): Matrix { return Matrix::quick([$this->a]); }
    public function asColumnMatrix(): Matrix
    { $a=[]; foreach($this->a as $v) $a[]= [$v]; return Matrix::quick($a); }
    public function flatten(): Vector { return Vector::quick($this->a); }

    // ── reductions ────────────────────────────────────────────────────────

    public function sum(): float   { return array_sum($this->a); }
    public function product(): float
    { $p=1.0; foreach($this->a as $v)$p*=$v; return $p; }
    public function min(): float   { return min($this->a); }
    public function max(): float   { return max($this->a); }
    public function mean(): float  { return array_sum($this->a) / $this->n; }

    public function median(): float
    {
        $s = $this->a; sort($s);
        return $this->n%2===0 ? ($s[$this->n/2-1]+$s[$this->n/2])/2.0 : $s[($this->n-1)/2];
    }

    public function variance(?Tensor $mean=null): float
    {
        $mu = $mean ? $mean->mean() : $this->mean();
        $s=0.0; foreach($this->a as $v) $s+=($v-$mu)**2;
        return $s/$this->n;
    }

    public function argmin(): int
    {
        $min=INF; $idx=0;
        foreach($this->a as $i=>$v) if($v<$min){$min=$v;$idx=$i;}
        return $idx;
    }

    public function argmax(): int
    {
        $max=-INF; $idx=0;
        foreach($this->a as $i=>$v) if($v>$max){$max=$v;$idx=$i;}
        return $idx;
    }

    // ── element-wise unary ops ────────────────────────────────────────────

    private function mapEl(callable $fn): static
    {
        $a=[]; foreach($this->a as $v) $a[]=$fn($v);
        return new static($a,$this->n);
    }

    public function round(int $precision=0): static { return $this->mapEl(fn($v)=>round($v,$precision)); }
    public function floor(): static  { return $this->mapEl('floor'); }
    public function ceil(): static   { return $this->mapEl('ceil'); }
    public function abs(): static    { return $this->mapEl('abs'); }
    public function square(): static { return $this->mapEl(fn($v)=>$v*$v); }
    public function sqrt(): static   { return $this->mapEl('sqrt'); }
    public function exp(): static    { return $this->mapEl('exp'); }
    public function expm1(): static  { return $this->mapEl('expm1'); }
    public function log(float $base=M_E): static
    { return $base===M_E?$this->mapEl('log'):$this->mapEl(fn($v)=>log($v,$base)); }
    public function log1p(): static  { return $this->mapEl('log1p'); }
    public function sin(): static    { return $this->mapEl('sin'); }
    public function cos(): static    { return $this->mapEl('cos'); }
    public function tan(): static    { return $this->mapEl('tan'); }
    public function asin(): static   { return $this->mapEl('asin'); }
    public function acos(): static   { return $this->mapEl('acos'); }
    public function atan(): static   { return $this->mapEl('atan'); }
    public function sinh(): static   { return $this->mapEl('sinh'); }
    public function cosh(): static   { return $this->mapEl('cosh'); }
    public function tanh(): static   { return $this->mapEl('tanh'); }
    public function negate(): static { return $this->mapEl(fn($v)=>-$v); }
    public function reciprocal(): static { return $this->mapEl(fn($v)=>1.0/$v); }
    public function map(callable $fn): static { return $this->mapEl($fn); }

    public function clip(float $min,float $max): static
    { return $this->mapEl(fn($v)=>max($min,min($max,$v))); }
    public function clipLower(float $min): static { return $this->mapEl(fn($v)=>max($min,$v)); }
    public function clipUpper(float $max): static { return $this->mapEl(fn($v)=>min($max,$v)); }

    public function reduce(callable $fn, int|float $initial=0): int|float
    { $acc=$initial; foreach($this->a as $v) $acc=$fn($acc,$v); return $acc; }

    public function asType(int $dtype): static
    {
        $fn=$dtype===Tensor::INT?fn($v)=>(float)(int)$v:fn($v)=>(float)$v;
        return $this->mapEl($fn);
    }

    // ── arithmetic ────────────────────────────────────────────────────────

    public function add(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v+(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$v+$b->a[$i];
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function subtract(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v-(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$v-$b->a[$i];
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function multiply(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v*(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$v*$b->a[$i];
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function divide(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v/(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$v/$b->a[$i];
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function pow(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$v**(float)$b);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=pow($v,$b->a[$i]);
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function mod(int|float|Tensor $b): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>fmod($v,(float)$b));
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=fmod($v,$b->a[$i]);
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    private function cmp(int|float|Tensor $b, callable $fn): static
    {
        if (is_numeric($b)) return $this->mapEl(fn($v)=>$fn($v,(float)$b)?1.0:0.0);
        if ($b instanceof static && $b->n===$this->n) {
            $a=[]; foreach($this->a as $i=>$v) $a[]=$fn($v,$b->a[$i])?1.0:0.0;
            return new static($a,$this->n);
        }
        throw new \InvalidArgumentException("Incompatible operand");
    }

    public function equal(int|float|Tensor $b): static       { return $this->cmp($b,fn($x,$y)=>$x==$y); }
    public function notEqual(int|float|Tensor $b): static    { return $this->cmp($b,fn($x,$y)=>$x!=$y); }
    public function greater(int|float|Tensor $b): static     { return $this->cmp($b,fn($x,$y)=>$x>$y); }
    public function greaterEqual(int|float|Tensor $b): static{ return $this->cmp($b,fn($x,$y)=>$x>=$y); }
    public function less(int|float|Tensor $b): static        { return $this->cmp($b,fn($x,$y)=>$x<$y); }
    public function lessEqual(int|float|Tensor $b): static   { return $this->cmp($b,fn($x,$y)=>$x<=$y); }

    // ── vector-specific ops ───────────────────────────────────────────────

    /** Dot product with another vector */
    public function dot(Vector $b): float
    {
        if ($b->n !== $this->n) throw new \RuntimeException("Length mismatch for dot product");
        $blas = Blas::getInstance();
        if ($blas->available()) return $blas->ddot($this->a, $b->a, $this->n);
        $s=0.0; foreach($this->a as $i=>$v) $s+=$v*$b->a[$i]; return $s;
    }

    public function inner(Vector $b): float { return $this->dot($b); }

    /** Outer product: this(n) × b(m) → Matrix(n×m) */
    public function outer(Vector $b): Matrix
    {
        $bArr = $b->asArray(); $m = $b->n(); $a=[];
        foreach($this->a as $i=>$vi) { $r=[]; foreach($bArr as $j=>$vj) $r[]=$vi*$vj; $a[$i]=$r; }
        return Matrix::quick($a);
    }

    /** 3D cross product */
    public function cross(Vector $b): static
    {
        if ($this->n!==3||$b->n!==3) throw new \RuntimeException("Cross product requires 3D vectors");
        $a=$this->a; $bA=$b->a;
        return new static([
            $a[1]*$bA[2]-$a[2]*$bA[1],
            $a[2]*$bA[0]-$a[0]*$bA[2],
            $a[0]*$bA[1]-$a[1]*$bA[0],
        ], 3);
    }

    /** p-norm */
    public function norm(float $p=2.0): float
    {
        if ($p===2.0) {
            $blas=Blas::getInstance();
            if ($blas->available()) return $blas->dnrm2($this->a,$this->n);
            $s=0.0; foreach($this->a as $v) $s+=$v*$v; return sqrt($s);
        }
        if ($p===1.0) { $s=0.0; foreach($this->a as $v) $s+=abs($v); return $s; }
        if (is_infinite($p)) { $m=0.0; foreach($this->a as $v) $m=max($m,abs($v)); return $m; }
        $s=0.0; foreach($this->a as $v) $s+=pow(abs($v),$p); return pow($s,1.0/$p);
    }

    /** Normalise to unit vector under p-norm */
    public function normalize(float $p=2.0): static
    {
        $n=$this->norm($p); if($n<1e-14) return clone $this;
        return $this->divide($n);
    }

    /** Vector projection of this onto b */
    public function project(Vector $b): static
    {
        $bn2 = $b->dot($b); if($bn2<1e-14) throw new \RuntimeException("Cannot project onto zero vector");
        return $b->multiply($this->dot($b)/$bn2);
    }

    /** Reflect this vector through the plane with normal b */
    public function reflect(Vector $b): static
    {
        return $this->subtract($b->multiply(2.0*$this->dot($b)/$b->dot($b)));
    }

    /** 1D convolution of this with kernel $b */
    public function convolve(Vector $b, int $stride=1): static
    {
        $kn=$b->n(); $on=(int)(($this->n-$kn)/$stride)+1;
        $bArr=$b->asArray(); $a=[];
        for($i=0;$i<$on;++$i){
            $s=0.0; for($k=0;$k<$kn;++$k) $s+=$this->a[$i*$stride+$k]*$bArr[$k];
            $a[]=$s;
        }
        return new static($a,$on);
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

    #[\Override]
    public function abs(): static {
        
    }

    #[\Override]
    public function acos(): static {
        
    }

    #[\Override]
    public function add(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function asArray(): array {
        
    }

    #[\Override]
    public function asin(): static {
        
    }

    #[\Override]
    public function atan(): static {
        
    }

    #[\Override]
    public function ceil(): static {
        
    }

    #[\Override]
    public function clip(float $min, float $max): static {
        
    }

    #[\Override]
    public function clipLower(float $min): static {
        
    }

    #[\Override]
    public function clipUpper(float $max): static {
        
    }

    #[\Override]
    public function cos(): static {
        
    }

    #[\Override]
    public function cosh(): static {
        
    }

    #[\Override]
    public function count(): int {
        
    }

    #[\Override]
    public function divide(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function dtype(): int {
        
    }

    #[\Override]
    public function equal(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function exp(): static {
        
    }

    #[\Override]
    public function expm1(): static {
        
    }

    #[\Override]
    public function flatten(): Vector {
        
    }

    #[\Override]
    public function floor(): static {
        
    }

    #[\Override]
    public function getIterator(): \Traversable {
        
    }

    #[\Override]
    public function greater(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function greaterEqual(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function isSquare(): bool {
        
    }

    #[\Override]
    public function jsonSerialize(): mixed {
        
    }

    #[\Override]
    public function less(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function lessEqual(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function log(float $base = M_E): static {
        
    }

    #[\Override]
    public function log1p(): static {
        
    }

    #[\Override]
    public function map(callable $fn): static {
        
    }

    #[\Override]
    public function max(): int|float {
        
    }

    #[\Override]
    public function mean(): float {
        
    }

    #[\Override]
    public function median(): float {
        
    }

    #[\Override]
    public function min(): int|float {
        
    }

    #[\Override]
    public function mod(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function multiply(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function ndim(): int {
        
    }

    #[\Override]
    public function negate(): static {
        
    }

    #[\Override]
    public function notEqual(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function pow(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function product(): int|float {
        
    }

    #[\Override]
    public function reciprocal(): static {
        
    }

    #[\Override]
    public function reduce(callable $fn, int|float $initial = 0): int|float {
        
    }

    #[\Override]
    public function round(int $precision = 0): static {
        
    }

    #[\Override]
    public function shape(): array {
        
    }

    #[\Override]
    public function shapeString(): string {
        
    }

    #[\Override]
    public function sin(): static {
        
    }

    #[\Override]
    public function sinh(): static {
        
    }

    #[\Override]
    public function size(): int {
        
    }

    #[\Override]
    public function sqrt(): static {
        
    }

    #[\Override]
    public function square(): static {
        
    }

    #[\Override]
    public function subtract(int|float|\Tensor $b): static {
        
    }

    #[\Override]
    public function sum(): int|float {
        
    }

    #[\Override]
    public function tan(): static {
        
    }

    #[\Override]
    public function tanh(): static {
        
    }

    #[\Override]
    public function variance(?\Tensor $mean = null): float {
        
    }
}