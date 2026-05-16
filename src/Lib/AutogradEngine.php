<?php
declare(strict_types=1);

namespace Pml\Lib;

/**
 * @deprecated Use TensorEngine::get() directly.
 *
 * AutogradEngine previously maintained its own FFI singleton that redeclared
 * TensorC, creating two incompatible FFI type universes when used alongside
 * TensorEngine.  The Tape / VarNode / ag_* / tape_* declarations have been
 * merged into TensorEngine's cdef block (§4/5/6 of the refactoring plan) so
 * that all TensorC* pointers live in one FFI universe and can be passed freely
 * between autograd ops and tensor ops.
 *
 * This class now returns TensorEngine::get() so existing call-sites keep
 * working without modification.  It will be removed in the next major version.
 */
final class AutogradEngine
{
    public static function get(): \FFI
    {
        trigger_error(
            'AutogradEngine::get() is deprecated — call TensorEngine::get() directly.',
            E_USER_DEPRECATED
        );
        return TensorEngine::get();
    }
}
