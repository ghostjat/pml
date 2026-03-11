<?php
declare(strict_types=1);

namespace Pml;


// ═══════════════════════════════════════════════════════════════════════════
//  COLUMN VECTOR CLASS
// ═══════════════════════════════════════════════════════════════════════════

class ColumnVector extends Vector
{
    public function asVector(): Vector { return Vector::quick($this->a); }
    public function asMatrix(): Matrix
    { $a=[]; foreach($this->a as $v) $a[]= [$v]; return Matrix::quick($a); }

    public function asRowMatrix(): Matrix
    { return Matrix::quick([$this->a]); }

    public function asColumnMatrix(): Matrix
    { $a=[]; foreach($this->a as $v) $a[]= [$v]; return Matrix::quick($a); }
}
