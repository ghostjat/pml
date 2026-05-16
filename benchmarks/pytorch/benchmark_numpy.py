#!/usr/bin/env python3
"""
NumPy CPU Benchmarks — PML comparison baseline.

Methodology:
  - 3 warmup iterations discarded
  - N measurement iterations, report median
  - Thread count matches OMP_NUM_THREADS env var
  - No JIT / tracing (NumPy uses MKL/OpenBLAS via standard Python calls)

Run:
  OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 python3 benchmarks/pytorch/benchmark_numpy.py
  python3 benchmarks/pytorch/benchmark_numpy.py --threads 1 --json > results/numpy_1t.json
  python3 benchmarks/pytorch/benchmark_numpy.py --threads 16 --json > results/numpy_16t.json
"""

import argparse
import json
import math
import os
import statistics
import time
from datetime import datetime

import numpy as np


def configure_threads(n: int) -> None:
    """Set thread count before importing BLAS-linked code."""
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[var] = str(n)


WARMUP_ITERS  = 3
BENCH_ITERS   = 10
ELEMENTS_1M   = 1_000_000
ELEMENTS_10M  = 10_000_000


def bench(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Run fn with warmup, return dict with median/mean/stdev/min/max in seconds."""
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - t0)
    return {
        'median_s':  statistics.median(timings),
        'mean_s':    statistics.mean(timings),
        'stdev_s':   statistics.stdev(timings) if len(timings) > 1 else 0.0,
        'min_s':     min(timings),
        'max_s':     max(timings),
        'rstdev_pct': (statistics.stdev(timings) / statistics.mean(timings) * 100)
                       if statistics.mean(timings) > 0 else 0.0,
        'iters':     iters,
    }


def gflops(n, time_s, iters=1):
    """GFLOPS for N×N × N×N matmul."""
    ops = 2 * n**3 * iters
    return ops / 1e9 / time_s if time_s > 0 else 0.0


class NumpyBenchmarks:
    def __init__(self, threads: int):
        self.threads  = threads
        self.results  = {}
        self.metadata = {
            'framework':    'numpy',
            'version':      np.__version__,
            'numpy_config': str(np.show_config(mode='dicts')).replace('\n', ''),
            'threads':      threads,
            'warmup_iters': WARMUP_ITERS,
            'bench_iters':  BENCH_ITERS,
            'timestamp':    datetime.now().isoformat(),
        }

    def run_all(self, quiet=False):
        sections = [
            ('gemm',         self._gemm),
            ('elementwise',  self._elementwise),
            ('reductions',   self._reductions),
            ('activations',  self._activations),
            ('linalg',       self._linalg),
            ('shape',        self._shape),
            ('memory_bw',    self._memory_bw),
        ]
        for name, fn in sections:
            if not quiet:
                print(f'[{name}] running...')
            fn()
        return self

    # ── GEMM ──────────────────────────────────────────────────────────────────

    def _gemm(self):
        for n in (64, 128, 256, 512, 1024, 2048):
            A = np.random.randn(n, n).astype(np.float32)
            B = np.random.randn(n, n).astype(np.float32)

            r = bench(lambda A=A, B=B: np.dot(A, B))

            self.results[f'gemm_{n}x{n}'] = {
                **r,
                'gflops': gflops(n, r['median_s']),
                'shape':  f'{n}x{n}',
            }

    # ── Element-wise ──────────────────────────────────────────────────────────

    def _elementwise(self):
        a = np.random.randn(ELEMENTS_1M).astype(np.float32)
        b = np.random.randn(ELEMENTS_1M).astype(np.float32)

        ops = {
            'add_1M':      lambda: np.add(a, b, out=np.empty_like(a)),
            'mul_1M':      lambda: np.multiply(a, b, out=np.empty_like(a)),
            'div_1M':      lambda: np.divide(a, b + 1e-6, out=np.empty_like(a)),
            'add_inplace_1M': lambda: (c := a.copy(), np.add(c, b, out=c)),
        }
        for name, fn in ops.items():
            r = bench(fn)
            gb_s = (ELEMENTS_1M * 4 * 3) / 1e9 / r['median_s']  # 3 arrays (a, b, out)
            self.results[name] = {**r, 'gb_s': round(gb_s, 2)}

    # ── Reductions ────────────────────────────────────────────────────────────

    def _reductions(self):
        a = np.random.randn(ELEMENTS_1M).astype(np.float32)
        m = np.random.randn(512, 512).astype(np.float32)

        self.results['sum_1M']        = bench(lambda: np.sum(a))
        self.results['mean_1M']       = bench(lambda: np.mean(a))
        self.results['max_1M']        = bench(lambda: np.max(a))
        self.results['std_1M']        = bench(lambda: np.std(a))
        self.results['sum_axis0_512'] = bench(lambda: np.sum(m, axis=0))
        self.results['sum_axis1_512'] = bench(lambda: np.sum(m, axis=1))

    # ── Activations ───────────────────────────────────────────────────────────

    def _activations(self):
        a = np.random.randn(ELEMENTS_1M).astype(np.float32)
        pos = np.abs(a) + 1e-4

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        def gelu(x):
            return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))

        self.results['relu_1M']    = bench(lambda: np.maximum(a, 0))
        self.results['sigmoid_1M'] = bench(lambda: sigmoid(a))
        self.results['tanh_1M']    = bench(lambda: np.tanh(a))
        self.results['exp_1M']     = bench(lambda: np.exp(a))
        self.results['sqrt_1M']    = bench(lambda: np.sqrt(pos))
        self.results['log_1M']     = bench(lambda: np.log(pos))

    # ── Linear algebra ────────────────────────────────────────────────────────

    def _linalg(self):
        m100  = np.random.randn(100, 100).astype(np.float32)
        m256  = np.random.randn(256, 256).astype(np.float32)
        m512  = np.random.randn(512, 512).astype(np.float32)

        self.results['matmul_256']   = bench(lambda: np.dot(m256, m256))
        self.results['matmul_512']   = bench(lambda: np.dot(m512, m512))
        self.results['svd_100']      = bench(lambda: np.linalg.svd(m100, full_matrices=False))
        self.results['inv_100']      = bench(lambda: np.linalg.inv(m100))
        self.results['lstsq_100']    = bench(lambda: np.linalg.lstsq(m100, np.ones((100, 1), dtype=np.float32), rcond=None))

    # ── Shape ops ─────────────────────────────────────────────────────────────

    def _shape(self):
        m = np.random.randn(512, 512).astype(np.float32)
        v = np.random.randn(ELEMENTS_1M).astype(np.float32)

        # Note: NumPy transpose is O(1) (returns a view), copy is O(N)
        self.results['transpose_512_view'] = bench(lambda: m.T)
        self.results['transpose_512_copy'] = bench(lambda: m.T.copy())
        self.results['reshape_1M']         = bench(lambda: v.reshape(1000, 1000))
        self.results['flatten_512']        = bench(lambda: m.flatten())
        self.results['copy_512']           = bench(lambda: m.copy())

    # ── Memory bandwidth ──────────────────────────────────────────────────────

    def _memory_bw(self):
        # STREAM-style: measure effective memory bandwidth
        n = ELEMENTS_10M
        a = np.ones(n, dtype=np.float32)
        b = np.ones(n, dtype=np.float32)
        c = np.empty(n, dtype=np.float32)

        # STREAM Copy: C = A
        r = bench(lambda: np.copyto(c, a))
        gb_s = (n * 4 * 2) / 1e9 / r['median_s']   # read A + write C
        self.results['stream_copy_10M'] = {**r, 'gb_s': round(gb_s, 2)}

        # STREAM Triad: C = A + scalar * B
        r = bench(lambda: np.add(a, 2.0 * b, out=c))
        gb_s = (n * 4 * 3) / 1e9 / r['median_s']
        self.results['stream_triad_10M'] = {**r, 'gb_s': round(gb_s, 2)}

    # ── Output ────────────────────────────────────────────────────────────────

    def print_summary(self):
        print('\n' + '='*60)
        print(f'NumPy {np.__version__} Benchmarks — {self.threads} threads')
        print('='*60)
        for name, r in self.results.items():
            if isinstance(r, dict) and 'median_s' in r:
                ms = r['median_s'] * 1000
                rst = r.get('rstdev_pct', 0)
                extra = ''
                if 'gflops' in r:
                    extra = f'  {r["gflops"]:.1f} GFLOPS'
                elif 'gb_s' in r:
                    extra = f'  {r["gb_s"]:.1f} GB/s'
                print(f'  {name:<30s}  {ms:8.3f} ms  rstdev={rst:.1f}%{extra}')
        print()

    def to_json(self) -> str:
        return json.dumps({'metadata': self.metadata, 'benchmarks': self.results},
                          indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description='NumPy CPU benchmarks for PML comparison')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', 1)))
    parser.add_argument('--json', action='store_true', help='Output JSON to stdout')
    parser.add_argument('--quiet', action='store_true', help='Suppress progress output')
    parser.add_argument('--iters', type=int, default=BENCH_ITERS)
    args = parser.parse_args()

    configure_threads(args.threads)
    global BENCH_ITERS
    BENCH_ITERS = args.iters

    bench_suite = NumpyBenchmarks(threads=args.threads)
    bench_suite.run_all(quiet=args.quiet)

    if args.json:
        print(bench_suite.to_json())
    else:
        bench_suite.print_summary()


if __name__ == '__main__':
    main()
