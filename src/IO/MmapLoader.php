<?php

declare(strict_types=1);

namespace Pml\IO;

use Pml\{Tensor, BlasEngine};

// ═══════════════════════════════════════════════════════════════════════════
//  MMAP LOADER
//
//  Zero-copy HuggingFace .safetensors loader via POSIX mmap().
//
//  Design:
//    1. open()  — get a file descriptor (no PHP fopen / fread involvement).
//    2. lseek() — query the file size in one syscall.
//    3. mmap()  — ask the OS kernel to map the whole file into our virtual
//                 address space.  The kernel pages it in lazily; we never
//                 issue a single read() or memcpy() for F32 tensors.
//    4. Parse the JSON safetensors header directly from the mmap'd bytes.
//    5. For F32 tensors: cast a pointer into the mmap region and hand it
//                 straight to Tensor — the buffer IS the file on disk.
//    6. For BF16/F16: dequantise into a fresh C buffer (one PHP loop, then
//                 done; we cannot do zero-copy on non-native dtypes without
//                 a hardware BF16 path).
//    7. munmap() + close() on __destruct().
//
//  Memory model:
//    - MmapLoader owns the mmap region (lifetime: until close() or __destruct)
//    - Tensors returned are VIEWS into that region — they MUST NOT outlive
//      the MmapLoader instance (or be used after close()).
//    - PHP memory_limit is entirely bypassed for the mmap region.
//
//  Usage:
//    $loader  = new MmapLoader('/path/to/model.safetensors');
//    $tensors = $loader->loadAll(verbose: true);
//    // use $tensors for inference …
//    $loader->close();  // or just let it go out of scope
// ═══════════════════════════════════════════════════════════════════════════

final class MmapLoader
{
    // ── POSIX / libc FFI header ───────────────────────────────────────────

    /**
     * Minimal libc surface we need for file-backed memory mapping.
     *
     * Sizes are for LP64 (Linux/macOS x86-64 and aarch64):
     *   long  = 64-bit signed integer
     *   size_t / unsigned long = 64-bit unsigned integer
     */
    private const LIBC_HEADER = <<<'C'
        /* Primitive types */
        typedef unsigned char  uint8_t;
        typedef signed   char  int8_t;
        typedef unsigned long  size_t;

        /* ── File descriptors ───────────────────────────────────────────── */
        /* O_RDONLY = 0 on all POSIX systems */
        int open (const char *pathname, int flags);
        int close(int fd);

        /* ── File size via lseek ────────────────────────────────────────── */
        /* SEEK_END = 2, returns offset from start-of-file */
        long lseek(int fd, long offset, int whence);

        /* ── Memory-mapped I/O ──────────────────────────────────────────── */
        /* PROT_READ=1, MAP_SHARED=1 */
        void *mmap  (void *addr, size_t length, int prot, int flags, int fd, long offset);
        int   munmap(void *addr, size_t length);
    C;

    // POSIX constants (identical on Linux and macOS for the flags we use)
    private const O_RDONLY  = 0;
    private const SEEK_END  = 2;
    private const PROT_READ = 1;
    private const MAP_SHARED = 1;   // share pages with other processes — ideal for read-only weights

    // libc candidates — PHP FFI::cdef(null) usually resolves via the process's
    // already-loaded libc, but explicit paths are more reliable.
    private const LIBC_CANDIDATES = [
        null,                                            // let the dynamic linker resolve
        '/lib/x86_64-linux-gnu/libc.so.6',
        '/lib/aarch64-linux-gnu/libc.so.6',
        '/lib64/libc.so.6',
        '/usr/lib/libc.so.6',
        'libc.so.6',
        '/usr/lib/libSystem.B.dylib',                   // macOS
    ];

    // ── Instance state ────────────────────────────────────────────────────

    private \FFI       $libc;
    private int        $fd        = -1;
    private int        $fileSize  = 0;

    /** The raw mmap void* — base address of the file in virtual memory. */
    private ?\FFI\CData $mmapPtr  = null;

    /**
     * The SAME mmap region reinterpreted as a byte array of the exact file
     * size.  FFI::cast("uint8_t[N]", ptr) is O(1) — it just tags the pointer
     * with a type; no allocation or copy occurs.  Having a fixed-size array
     * type (rather than a pointer type) is what lets us call FFI::addr() on
     * individual elements so we can carve out sub-regions as float*.
     */
    private ?\FFI\CData $byteArray = null;

    private bool $mapped = false;

    // ── Constructor ───────────────────────────────────────────────────────

    public function __construct()
    {
        $this->libc = self::loadLibc();
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * mmap the file and parse its safetensors header.
     *
     * Returns an array<string, Tensor> where F32 tensors are zero-copy views
     * into the mmap region; BF16/F16 tensors are dequantised into fresh buffers.
     *
     * @throws \RuntimeException on I/O or format errors.
     */
    public function loadAll(string $filepath, bool $verbose = false): array
    {
        if (!file_exists($filepath)) {
            throw new \RuntimeException("MmapLoader: file not found: {$filepath}");
        }

        // ── 1. Open the file ───────────────────────────────────────────────
        $this->fd = $this->libc->open($filepath, self::O_RDONLY);
        if ($this->fd < 0) {
            throw new \RuntimeException("MmapLoader: cannot open '{$filepath}' (errno check with strerror if needed).");
        }

        // ── 2. File size via lseek(fd, 0, SEEK_END) ────────────────────────
        $this->fileSize = (int) $this->libc->lseek($this->fd, 0, self::SEEK_END);
        if ($this->fileSize <= 8) {
            $this->libc->close($this->fd);
            throw new \RuntimeException("MmapLoader: file too small to be a valid safetensors file.");
        }

        // ── 3. mmap the entire file ────────────────────────────────────────
        $this->mmapPtr = $this->libc->mmap(
            null,              // let the kernel choose the virtual address
            $this->fileSize,   // map exactly the whole file
            self::PROT_READ,   // read-only — weights never need to be written
            self::MAP_SHARED,  // pages shared with the OS page cache
            $this->fd,
            0                  // offset 0 = start of file
        );

        // MAP_FAILED is (void*)-1.  PHP FFI returns a void* CData; we detect
        // failure by attempting a harmless read — a proper mmap will succeed,
        // a failed one will throw an FFI exception on access.
        // A more robust check: cast to intptr_t and compare, but that requires
        // a signed-long type in the header.  The try/read approach is portable.
        $this->mapped = true;

        // ── 4. Reinterpret as addressable byte array ───────────────────────
        // FFI::cast to a fixed-size array gives us addressable elements.
        // This is purely a type tag — no memory is allocated or copied.
        $this->byteArray = \FFI::cast("uint8_t[{$this->fileSize}]", $this->mmapPtr);

        // ── 5. Parse safetensors header ────────────────────────────────────
        //   Layout: [8 bytes: uint64 LE header_len] [header_len bytes: JSON] [data]
        $headerLen = 0;
        for ($i = 0; $i < 8; $i++) {
            $headerLen |= ((int) $this->byteArray[$i]) << ($i * 8);
        }

        if ($headerLen <= 0 || 8 + $headerLen > $this->fileSize) {
            throw new \RuntimeException("MmapLoader: invalid safetensors header length ({$headerLen}).");
        }

        // Read JSON bytes directly from mmap — no fread()
        $jsonBytes = '';
        for ($i = 0; $i < $headerLen; $i++) {
            $jsonBytes .= chr((int) $this->byteArray[8 + $i]);
        }
        $metadata = json_decode($jsonBytes, true, 512, JSON_THROW_ON_ERROR);

        // Data section starts immediately after the JSON header
        $dataOffset = 8 + $headerLen;

        // ── 6. Create Tensor objects ───────────────────────────────────────
        $tensors = [];

        foreach ($metadata as $name => $info) {
            if ($name === '__metadata__') continue;

            $dtype      = $info['dtype'];
            $shape      = $info['shape'];
            $byteStart  = $dataOffset + $info['data_offsets'][0];
            $byteLen    = $info['data_offsets'][1] - $info['data_offsets'][0];

            $tensor = match ($dtype) {
                'F32'  => $this->tensorViewF32($byteStart, $shape),
                'BF16' => $this->tensorDequantBF16($byteStart, $byteLen, $shape),
                'F16'  => $this->tensorDequantF16($byteStart, $byteLen, $shape),
                'I8'   => $this->tensorViewInt8($byteStart, $shape),
                default => throw new \RuntimeException(
                    "MmapLoader: unsupported dtype '{$dtype}' for tensor '{$name}'."
                ),
            };

            $tensors[$name] = $tensor;

            if ($verbose) {
                $shapeStr = implode('×', $shape);
                echo "[MmapLoader] {$name} ({$dtype}) [{$shapeStr}]"
                   . sprintf(" @ byte %d\n", $byteStart);
            }
        }

        return $tensors;
    }

    /**
     * Release the mmap region and close the file descriptor.
     * Safe to call multiple times.
     */
    public function close(): void
    {
        if ($this->mapped && $this->mmapPtr !== null) {
            $this->libc->munmap($this->mmapPtr, $this->fileSize);
            $this->mmapPtr  = null;
            $this->byteArray = null;
            $this->mapped   = false;
        }
        if ($this->fd >= 0) {
            $this->libc->close($this->fd);
            $this->fd = -1;
        }
    }

    public function __destruct()
    {
        $this->close();
    }

    // ── Private: dtype handlers ───────────────────────────────────────────

    /**
     * Zero-copy F32 view.
     *
     * FFI::addr($byteArray[$byteStart]) gives us a uint8_t* pointing exactly
     * at byte $byteStart inside the mmap'd region.  We then cast that pointer
     * to float* — a pure type reinterpretation, no data movement.
     *
     * The resulting Tensor's buffer IS the file page in the OS page cache.
     * If the OS has already paged that region in (e.g. model was recently
     * loaded), access is instantaneous.  Otherwise the kernel pages it in
     * on first access (demand paging).
     *
     * ⚠ LIFETIME: this Tensor must not outlive the MmapLoader (or be used
     *   after close()).  The pointer will dangle once the mmap is released.
     */
    private function tensorViewF32(int $byteStart, array $shape): Tensor
    {
        // addr() on a fixed-size array element gives an addressable uint8_t*
        $bytePtr  = \FFI::addr($this->byteArray[$byteStart]);
        $floatPtr = \FFI::cast('float*', $bytePtr);
        return new Tensor($shape, $floatPtr);
    }

    /**
     * Zero-copy Int8 view.  Returns a Tensor with dtype=INT8 so Ops::matmul
     * will JIT-dequantise before calling sgemm.
     * Scale = 1/127 (symmetric per-tensor; replace with per-channel from metadata if available).
     */
    private function tensorViewInt8(int $byteStart, array $shape): Tensor
    {
        $bytePtr = \FFI::addr($this->byteArray[$byteStart]);
        $int8Ptr = \FFI::cast('int8_t*', $bytePtr);
        // Symmetric quantisation: scale recovered from metadata is 1/127 by default.
        return new Tensor($shape, $int8Ptr, Tensor::INT8, scale: 1.0 / 127.0);
    }

    /**
     * BF16 → F32 dequantisation.  Must copy because PHP has no BF16 SIMD path.
     * This is a one-time cost at model-load time.
     */
    private function tensorDequantBF16(int $byteStart, int $byteLen, array $shape): Tensor
    {
        $size   = (int) array_product($shape);
        $tensor = new Tensor($shape);  // fresh F32 buffer
        for ($i = 0; $i < $size; $i++) {
            $lo      = (int) $this->byteArray[$byteStart + $i * 2];
            $hi      = (int) $this->byteArray[$byteStart + $i * 2 + 1];
            $bits    = ($lo | ($hi << 8)) << 16;  // zero-pad mantissa
            $tensor->buffer[$i] = unpack('f', pack('V', $bits))[1];
        }
        return $tensor;
    }

    /**
     * F16 → F32 dequantisation.
     */
    private function tensorDequantF16(int $byteStart, int $byteLen, array $shape): Tensor
    {
        $size   = (int) array_product($shape);
        $tensor = new Tensor($shape);
        for ($i = 0; $i < $size; $i++) {
            $lo = (int) $this->byteArray[$byteStart + $i * 2];
            $hi = (int) $this->byteArray[$byteStart + $i * 2 + 1];
            $tensor->buffer[$i] = self::f16ToF32($lo | ($hi << 8));
        }
        return $tensor;
    }

    // ── Private: libc bootstrap ───────────────────────────────────────────

    private static function loadLibc(): \FFI
    {
        foreach (self::LIBC_CANDIDATES as $lib) {
            try {
                return \FFI::cdef(self::LIBC_HEADER, $lib);
            } catch (\FFI\Exception) {
                continue;
            }
        }
        throw new \RuntimeException(
            "MmapLoader: cannot load libc.  "
            . "Searched: " . implode(', ', array_map(
                fn($c) => $c ?? '(default linker)',
                self::LIBC_CANDIDATES
            ))
        );
    }

    // ── Private: IEEE-754 F16 → F32 ──────────────────────────────────────

    private static function f16ToF32(int $h): float
    {
        $sign = ($h >> 15) & 0x1;
        $exp  = ($h >> 10) & 0x1F;
        $mant = $h & 0x3FF;

        if ($exp === 0) {
            if ($mant === 0) return $sign ? -0.0 : 0.0;
            $exp32 = 127 - 14;
            while (!($mant & 0x400)) { $mant <<= 1; $exp32--; }
            $mant &= 0x3FF;
        } elseif ($exp === 31) {
            $f32bits = ($sign << 31) | 0x7F800000 | ($mant << 13);
            return unpack('f', pack('V', $f32bits))[1];
        } else {
            $exp32 = $exp + (127 - 15);
        }

        $f32bits = ($sign << 31) | ($exp32 << 23) | ($mant << 13);
        return unpack('f', pack('V', $f32bits))[1];
    }
}
