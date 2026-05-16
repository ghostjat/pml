#!/usr/bin/env python3
"""
scikit-learn CPU Benchmarks — PML comparison baseline.

Tests end-to-end fit+predict pipelines on synthetic datasets matching
the workloads in benchmarks/Workloads/TabularMLBench.php.

Methodology:
  - Datasets generated with fixed random seed for reproducibility
  - Time measured over 3 iterations (fit is expensive; we measure it once
    and average predict over 10 iterations)
  - Thread count set via environment variables before import

Run:
  OMP_NUM_THREADS=16 python3 benchmarks/pytorch/benchmark_sklearn.py
  python3 benchmarks/pytorch/benchmark_sklearn.py --threads 1 --json
"""

import argparse
import json
import os
import statistics
import time
from datetime import datetime

import numpy as np


def configure_threads(n: int) -> None:
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'LOKY_MAX_CPU_COUNT'):
        os.environ[var] = str(n)


# Import after thread configuration
configure_threads(int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 1)))

from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn


RANDOM_SEED = 42
BENCH_ITERS = 3  # fit is expensive — 3 is sufficient


def time_fn(fn, iters=BENCH_ITERS):
    timings = []
    for _ in range(iters):
        t0 = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - t0)
    return {
        'median_s':   statistics.median(timings),
        'mean_s':     statistics.mean(timings),
        'stdev_s':    statistics.stdev(timings) if len(timings) > 1 else 0.0,
        'min_s':      min(timings),
        'iters':      iters,
    }, result


class SklearnBenchmarks:
    def __init__(self, threads: int):
        self.threads = threads
        self.results = {}
        self.metadata = {
            'framework': 'sklearn',
            'version':   sklearn.__version__,
            'threads':   threads,
            'timestamp': datetime.now().isoformat(),
        }

    def run_all(self, quiet=False):
        if not quiet:
            print('[sklearn] Generating synthetic datasets...')

        # Classification datasets
        X_clf_5k20, y_clf_5k20 = make_classification(
            n_samples=5000, n_features=20, n_classes=3, n_informative=10,
            random_state=RANDOM_SEED
        )
        X_clf_2k20, y_clf_2k20 = make_classification(
            n_samples=2000, n_features=20, n_classes=2, n_informative=8,
            random_state=RANDOM_SEED
        )
        X_clf_5k50, y_clf_5k50 = make_classification(
            n_samples=5000, n_features=50, n_classes=5, n_informative=20,
            random_state=RANDOM_SEED
        )

        # Regression datasets
        X_reg_5k10, y_reg_5k10 = make_regression(
            n_samples=5000, n_features=10, random_state=RANDOM_SEED
        )

        # Unlabeled for clustering/decomposition
        rng = np.random.default_rng(RANDOM_SEED)
        X_unlab_2k20 = rng.standard_normal((2000, 20)).astype(np.float32)

        # Predict dataset (1K samples, same features as training)
        X_pred_1k20 = rng.standard_normal((1000, 20)).astype(np.float32)
        X_pred_1k10 = rng.standard_normal((1000, 10)).astype(np.float32)

        benchmarks = [
            ('logistic_regression_5k20',
             lambda: self._bench_fit_predict(
                 LogisticRegression(max_iter=100, C=1.0, solver='lbfgs', random_state=RANDOM_SEED),
                 X_clf_5k20, y_clf_5k20, X_pred_1k20)),

            ('gaussian_nb_5k20',
             lambda: self._bench_fit_predict(
                 GaussianNB(),
                 X_clf_5k20, y_clf_5k20, X_pred_1k20)),

            ('random_forest_100_5k20',
             lambda: self._bench_fit_predict(
                 RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_SEED),
                 X_clf_5k20, y_clf_5k20, X_pred_1k20)),

            ('gbdt_200_5k20',
             lambda: self._bench_fit_predict(
                 GradientBoostingClassifier(n_estimators=200, max_depth=6, random_state=RANDOM_SEED),
                 X_clf_5k20, y_clf_5k20, X_pred_1k20)),

            ('knn_5_2k20',
             lambda: self._bench_fit_predict(
                 KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree'),
                 X_clf_2k20, y_clf_2k20, X_pred_1k20)),

            ('linear_regression_5k10',
             lambda: self._bench_fit_predict(
                 LinearRegression(),
                 X_reg_5k10, y_reg_5k10, X_pred_1k10)),

            ('ridge_5k10',
             lambda: self._bench_fit_predict(
                 Ridge(alpha=0.1),
                 X_reg_5k10, y_reg_5k10, X_pred_1k10)),

            ('kmeans_8_2k20',
             lambda: self._bench_unsupervised(
                 KMeans(n_clusters=8, n_init=3, max_iter=100, random_state=RANDOM_SEED),
                 X_unlab_2k20)),

            ('pca_10_2k20',
             lambda: self._bench_unsupervised(
                 PCA(n_components=10), X_unlab_2k20)),

            ('standard_scaler_5k20',
             lambda: self._bench_transformer(StandardScaler(), X_clf_5k20)),
        ]

        for name, fn in benchmarks:
            if not quiet:
                print(f'  [{name}]...')
            fn()

        return self

    def _bench_fit_predict(self, model, X_train, y_train, X_pred):
        # Fit timing
        fit_r, _ = time_fn(lambda: model.fit(X_train, y_train), iters=BENCH_ITERS)

        # Predict timing (10 iterations, fitted model)
        pred_r, _ = time_fn(lambda: model.predict(X_pred), iters=10)

        bench_name = type(model).__name__
        self.results[bench_name.lower() + '_fit']     = fit_r
        self.results[bench_name.lower() + '_predict'] = pred_r

        # Also store combined key that matches PHP benchmark naming
        key = f'fit_{X_train.shape[0]}s_{X_train.shape[1]}f'
        self.results[f'{bench_name.lower()}_{key}'] = {
            'fit_median_s':     fit_r['median_s'],
            'predict_median_s': pred_r['median_s'],
            'n_train':          X_train.shape[0],
            'n_features':       X_train.shape[1],
            'n_predict':        X_pred.shape[0],
        }

    def _bench_unsupervised(self, model, X):
        r, _ = time_fn(lambda: model.fit(X))
        self.results[type(model).__name__.lower() + '_fit'] = r

    def _bench_transformer(self, transformer, X):
        fit_r, _ = time_fn(lambda: transformer.fit(X))
        transform_r, _ = time_fn(lambda: transformer.fit_transform(X))
        name = type(transformer).__name__.lower()
        self.results[f'{name}_fit']       = fit_r
        self.results[f'{name}_transform'] = transform_r

    def print_summary(self):
        print('\n' + '='*60)
        print(f'scikit-learn {sklearn.__version__} — {self.threads} threads')
        print('='*60)
        for name, r in self.results.items():
            if isinstance(r, dict) and 'median_s' in r:
                ms = r['median_s'] * 1000
                std_ms = r.get('stdev_s', 0) * 1000
                print(f'  {name:<45s}  {ms:8.1f} ms  ±{std_ms:.1f} ms')
        print()

    def to_json(self) -> str:
        return json.dumps({'metadata': self.metadata, 'benchmarks': self.results},
                          indent=2, default=str)


def main():
    parser = argparse.ArgumentParser(description='scikit-learn CPU benchmarks for PML comparison')
    parser.add_argument('--threads', type=int,
                        default=int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 1)))
    parser.add_argument('--json',  action='store_true')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    configure_threads(args.threads)

    suite = SklearnBenchmarks(threads=args.threads)
    suite.run_all(quiet=args.quiet)

    if args.json:
        print(suite.to_json())
    else:
        suite.print_summary()


if __name__ == '__main__':
    main()
