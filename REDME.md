# Pml: High-Performance Machine Learning for PHP

![PHP Version](https://img.shields.io/badge/PHP-8.1%2B-blue)
![FFI](https://img.shields.io/badge/FFI-Enabled-brightgreen)
![OpenBLAS](https://img.shields.io/badge/OpenBLAS-Accelerated-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

**Pml** is a pure-PHP, enterprise-grade Machine Learning and Deep Learning framework. It bypasses PHP's native array limitations by using the Foreign Function Interface (FFI) to orchestrate raw C-memory (`Float32` Tensors) directly through **OpenBLAS**, **LibSVM**, and **XGBoost**.

Whether you want to build a scikit-learn style `Pipeline` for classic machine learning, or train a custom Multi-Layer Perceptron using a dynamic Autograd engine, Pml brings true, C-speed AI to the PHP backend.

---

## ⚡ Key Features

* **Scikit-Learn Parity:** `Pml\Classic` provides an identical API to Python's scikit-learn (`fit`, `predict`, `fit_transform`).
* **Blazing Fast Math:** Zero-copy 1D Float32 flat-buffers routed directly into `libopenblas.so` for hardware-accelerated matrix multiplication (`sgemm`).
* **Deep Learning Autograd:** A fully-featured dynamic computation graph, supporting backpropagation, Cross-Entropy Loss, and AdamW optimization.
* **Native Integrations:** Direct FFI bindings to industry-standard C libraries: `libsvm` and `libxgboost`.
* **Memory Safe:** Rigorously tested with Resident Set Size (RSS) trackers. Guaranteed no C-pointer memory leaks during garbage collection.
* **FFI-Safe Serialization:** Save and load models instantly using `Joblib` with bit-identical float preservation.

---

## 🚀 Installation

Pml requires PHP 8.1+ with the FFI extension enabled, and underlying OS-level C libraries.

**1. Install System Dependencies (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install libopenblas-dev libsvm-dev libxgboost-dev
```

**2. Enable FFI in your `php.ini`:**
```ini
ffi.enable=true
```

**3. Install via Composer:**
```bash 
composer require ghostjat/pml
```

---

## 📖 Quick Start

### Classic ML: The Pipeline API
Clean data, scale features, and train a Support Vector Machine in 5 lines of code.

```php
use Pml\Classic\Pipeline\make_pipeline;
use Pml\Classic\Preprocess\StandardScaler;
use Pml\Classic\Preprocess\SimpleImputer;
use Pml\Classic\SVM\SVC;

// 1. Build the pipeline
$pipeline = make_pipeline(
    new SimpleImputer(strategy: 'mean'),
    new StandardScaler(),
    new SVC(kernel: 'rbf', C: 1.0)
);

// 2. Train on Tensors
$pipeline->fit($X_train, $y_train);

// 3. Predict
$predictions = $pipeline->predict($X_test);
```

### Deep Learning: The Autograd Engine
Train a Neural Network using AdamW.

```php
use Pml\Classic\NeuralNetwork\MLPClassifier;

$model = new MLPClassifier(
    hidden_layer_sizes: [64, 32],
    activation: 'relu',
    solver: 'adamw',
    learning_rate: 1e-3,
    max_iter: 200
);

$model->fit($X_train, $y_train);
```

---

## 🧰 Supported Algorithms

**Linear Models**
* `LogisticRegression` (Binary & Multinomial Auto-routing)
* `Ridge`, `Lasso`, `ElasticNet`, `LinearRegression`

**Ensembles & Trees**
* `RandomForestClassifier`, `DecisionTreeClassifier`
* `AdaBoostClassifier`, `BaggingClassifier`
* `XGBClassifier` (via libxgboost)

**Support Vector Machines**
* `SVC`, `SVR` (via libsvm)

**Clustering & Distance**
* `KMeans`
* `DBSCAN`
* `KNeighborsClassifier`

**Preprocessing & Tuning**
* `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder`
* `PCA`, `VarianceThreshold`
* `GridSearchCV`, `cross_val_score`, `Pipeline`

---

## 🚧 Limitations

* **Float32 Only:** To maximize BLAS throughput and keep FFI footprints small, all internal tensors operate strictly on 32-bit floats.
* **CPU Bound:** Currently, matrix operations are handled by CPU-based OpenBLAS.
* **Data Size:** While highly optimized, datasets larger than available RAM will require chunked batching, as Pml currently loads Tensors entirely into memory.

---

## 🔮 Roadmap / Future To-Do

We are constantly pushing the boundaries of what PHP can do. Upcoming features include:

- [ ] **GPU Acceleration:** Implement `libcudart` and `cublas` bindings for NVIDIA GPU support.
- [ ] **DataFrames:** Integrate Apache Arrow or Polars FFI bindings to replace standard PHP arrays for data loading.
- [ ] **Advanced Metrics:** Expand the `Metrics` module to include ROC-AUC, F1-Score, and Confusion Matrix generation.
- [ ] **Model Checkpointing:** Epoch-based save states for the Deep Learning Autograd engine.

---

## 🤝 Contributing

Contributions are welcome! Please ensure that all pull requests pass the rigorous memory-leak test suite.

```bash
php tests/run_e2e.php
```

---

## 📄 License

Pml is open-sourced software licensed under the **[MIT license](LICENSE)**.