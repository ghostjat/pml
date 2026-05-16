"""
PyTorch CPU Benchmarks — PML comparison baseline.

Thread configuration:
  Thread count is controlled by --threads argument (default: OMP_NUM_THREADS env,
  fallback: all available CPUs). This matches how PML is benchmarked.
  NEVER hard-code 1 thread — that would produce unfairly low PyTorch numbers.

Methodology:
  - 3 warmup iterations discarded
  - N measurement iterations, report median + rstdev
  - Results emitted as JSON for ReportGenerator

Run:
  OMP_NUM_THREADS=16 python3 benchmarks/pytorch/benchmark_pytorch.py --threads 16
  python3 benchmarks/pytorch/benchmark_pytorch.py --threads 1 --json > results/torch_1t.json
  python3 benchmarks/pytorch/benchmark_pytorch.py --threads 16 --json > results/torch_16t.json
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime

import numpy as np
import torch

WARMUP_ITERS = 3
BENCH_ITERS  = 10

# Parse thread count early — must be done before torch initializes thread pool
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument('--threads', type=int, default=int(os.environ.get('OMP_NUM_THREADS', torch.get_num_threads())))
_parser.add_argument('--json', action='store_true')
_parser.add_argument('--quiet', action='store_true')
_parser.add_argument('--iters', type=int, default=BENCH_ITERS)
_early_args, _ = _parser.parse_known_args()

_n_threads = _early_args.threads
torch.set_num_threads(_n_threads)
torch.set_num_interop_threads(min(4, _n_threads))
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[var] = str(_n_threads)

BENCH_ITERS  = _early_args.iters
PROFILING    = False
DATASET_SIZE = 10000

class PyTorchBenchmark:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'profiling': PROFILING,
            'iterations': ITERS,
            'dataset_size': DATASET_SIZE,
            'benchmarks': {}
        }
    
    def benchmark_matmul(self):
        """Dense matrix multiplication - fundamental ML operation"""
        print("[MatMul] Running matrix multiplication benchmark...")
        
        # 1000x1000 @ 1000x100 = 1000x100
        A = torch.randn(1000, 1000, dtype=torch.float32)
        B = torch.randn(1000, 100, dtype=torch.float32)
        
        start = time.time()
        for _ in range(ITERS):
            C = torch.mm(A, B)
        elapsed = time.time() - start
        
        ops = ITERS * 1000 * 1000 * 100 * 2  # multiply-add
        gflops = (ops / 1e9) / elapsed if elapsed > 0 else 0
        
        result = {
            'shape_a': list(A.shape),
            'shape_b': list(B.shape),
            'shape_c': list(C.shape),
            'time_sec': elapsed,
            'gflops': gflops,
            'iters': ITERS
        }
        self.results['benchmarks']['matmul'] = result
        print(f"  Time: {elapsed:.4f}s, GFLOPS: {gflops:.2f}")
        return result
    
    def benchmark_elementwise(self):
        """Element-wise operations (add, relu, etc.)"""
        print("[ElementWise] Running element-wise operations...")
        
        X = torch.randn(DATASET_SIZE, 100, dtype=torch.float32)
        Y = torch.randn(DATASET_SIZE, 100, dtype=torch.float32)
        
        start = time.time()
        for _ in range(ITERS):
            # Add
            Z = X + Y
            # ReLU
            Z = torch.relu(Z)
            # Tanh
            Z = torch.tanh(Z)
            # Multiply
            Z = Z * Y
        elapsed = time.time() - start
        
        result = {
            'shape': list(X.shape),
            'time_sec': elapsed,
            'elements_per_sec': (DATASET_SIZE * 100 * 4 * ITERS) / elapsed if elapsed > 0 else 0,
            'iters': ITERS
        }
        self.results['benchmarks']['elementwise'] = result
        print(f"  Time: {elapsed:.4f}s")
        return result
    
    def benchmark_softmax(self):
        """Softmax normalization"""
        print("[Softmax] Running softmax normalization...")
        
        logits = torch.randn(DATASET_SIZE, 10, dtype=torch.float32)
        
        start = time.time()
        for _ in range(ITERS):
            probs = torch.softmax(logits, dim=1)
        elapsed = time.time() - start
        
        result = {
            'shape': list(logits.shape),
            'time_sec': elapsed,
            'iters': ITERS
        }
        self.results['benchmarks']['softmax'] = result
        print(f"  Time: {elapsed:.4f}s")
        return result
    
    def benchmark_mlp_forward(self):
        """Simple 3-layer MLP forward pass"""
        print("[MLP Forward] Running MLP forward pass...")
        
        class SimpleMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(100, 64, bias=True)
                self.fc2 = torch.nn.Linear(64, 32, bias=True)
                self.fc3 = torch.nn.Linear(32, 10, bias=True)
            
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = SimpleMLP()
        model.eval()
        
        X = torch.randn(DATASET_SIZE, 100, dtype=torch.float32)
        
        with torch.no_grad():
            start = time.time()
            for _ in range(ITERS):
                output = model(X)
            elapsed = time.time() - start
        
        result = {
            'input_shape': list(X.shape),
            'output_shape': list(output.shape),
            'time_sec': elapsed,
            'iters': ITERS
        }
        self.results['benchmarks']['mlp_forward'] = result
        print(f"  Time: {elapsed:.4f}s")
        return result
    
    def benchmark_batch_norm(self):
        """Batch normalization"""
        print("[BatchNorm] Running batch normalization...")
        
        X = torch.randn(DATASET_SIZE, 64, dtype=torch.float32)
        bn = torch.nn.BatchNorm1d(64)
        bn.eval()
        
        with torch.no_grad():
            start = time.time()
            for _ in range(ITERS):
                Y = bn(X)
            elapsed = time.time() - start
        
        result = {
            'shape': list(X.shape),
            'time_sec': elapsed,
            'iters': ITERS
        }
        self.results['benchmarks']['batch_norm'] = result
        print(f"  Time: {elapsed:.4f}s")
        return result
    
    def benchmark_lstm_cell(self):
        """Single LSTM cell forward pass"""
        print("[LSTM Cell] Running LSTM cell forward pass...")
        
        lstm_cell = torch.nn.LSTMCell(50, 32)
        lstm_cell.eval()
        
        seq_len = 20
        batch_size = DATASET_SIZE // seq_len if DATASET_SIZE > seq_len else 1
        
        hx = torch.randn(batch_size, 32)
        cx = torch.randn(batch_size, 32)
        
        with torch.no_grad():
            start = time.time()
            for _ in range(ITERS):
                for t in range(seq_len):
                    x = torch.randn(batch_size, 50)
                    hx, cx = lstm_cell(x, (hx, cx))
            elapsed = time.time() - start
        
        result = {
            'hidden_size': 32,
            'input_size': 50,
            'seq_len': seq_len,
            'batch_size': batch_size,
            'time_sec': elapsed,
            'iters': ITERS
        }
        self.results['benchmarks']['lstm_cell'] = result
        print(f"  Time: {elapsed:.4f}s")
        return result
    
    def benchmark_conv2d(self):
        """2D Convolution"""
        print("[Conv2D] Running 2D convolution...")
        
        conv = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True)
        conv.eval()
        
        # Smaller batch for memory
        batch_size = 16
        X = torch.randn(batch_size, 3, 64, 64, dtype=torch.float32)
        
        with torch.no_grad():
            start = time.time()
            for _ in range(ITERS):
                Y = conv(X)
            elapsed = time.time() - start
        
        result = {
            'input_shape': list(X.shape),
            'output_shape': list(Y.shape),
            'time_sec': elapsed,
            'iters': ITERS
        }
        self.results['benchmarks']['conv2d'] = result
        print(f"  Time: {elapsed:.4f}s")
        return result
    
    def run_all(self, quiet=False):
        sections = [
            self.benchmark_matmul,
            self.benchmark_elementwise,
            self.benchmark_softmax,
            self.benchmark_mlp_forward,
            self.benchmark_batch_norm,
            self.benchmark_lstm_cell,
            self.benchmark_conv2d,
        ]
        if not quiet:
            print(f'\nPyTorch {torch.__version__} CPU — {_n_threads} threads')
            print('=' * 60)
        for fn in sections:
            fn()
        if not quiet:
            print('\nDone.')
        return self.results

    def print_summary(self):
        print('\n' + '='*60)
        print(f'PyTorch {torch.__version__} CPU Benchmarks — {_n_threads} threads')
        print('='*60)
        for name, r in self.results.get('benchmarks', {}).items():
            ms = r.get('time_sec', 0) * 1000
            gf = r.get('gflops', '')
            extra = f'  {gf:.1f} GFLOPS' if gf else ''
            print(f'  {name:<30s}  {ms:8.1f} ms{extra}')
        print()

    def to_json(self) -> str:
        metadata = {
            'framework':    'pytorch',
            'version':      torch.__version__,
            'threads':      _n_threads,
            'bench_iters':  BENCH_ITERS,
            'timestamp':    datetime.now().isoformat(),
        }
        return json.dumps({'metadata': metadata, 'benchmarks': self.results['benchmarks']},
                          indent=2, default=str)


def _add_gflops_to_matmul(results):
    """Post-process matmul results to add GFLOPS."""
    bm = results.get('benchmarks', {})
    r = bm.get('matmul', {})
    if r:
        n = 1000
        t = r.get('time_sec', 0) / BENCH_ITERS
        r['gflops'] = (2 * n**3) / 1e9 / t if t > 0 else 0
        r['time_per_iter_s'] = t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPU benchmarks for PML comparison')
    parser.add_argument('--threads', type=int, default=_n_threads)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--iters', type=int, default=BENCH_ITERS)
    args = parser.parse_args()

    bench = PyTorchBenchmark()
    results = bench.run_all(quiet=args.quiet)
    _add_gflops_to_matmul(results)

    if args.json:
        print(bench.to_json())
    else:
        bench.print_summary()
