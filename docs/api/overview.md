# API Overview

This framework exposes a PHP-native API for data management, tensor arithmetic, model training, and persistence.

## Main packages

- `Pml\Dataset` — mixed-type CSV ETL plus tensor-backed numeric datasets.
- `Pml\Tensor` — native tensor algebra, views, slicing, and BLAS-backed operations.
- `Pml\Pipeline` — transformer + estimator orchestration.
- `Pml\Training\Trainer` — high-level training loop with logging and checkpointing.
- `Pml\Lib\ModelStore` — safe configuration and tensor persistence.
- `Pml\Lib\SafeTensorsIO` — SafeTensors serialization and mmap-backed loading.

## Interface contracts

- `Pml\Interfaces\Estimator` — all models implement `predict(Dataset $dataset): Tensor`.
- `Pml\Interfaces\Learner` — models that support `train(Dataset $dataset): void`.
- `Pml\Interfaces\TrainableWithOptions` — models that accept extra training args.
- `Pml\Interfaces\Persistable` — save/load model state without PHP `serialize()`.
- `Pml\Interfaces\Stateful` — exposes tensor state for zero-copy persistence.

## Model categories

The `src/Estimators/` directory contains these families:

- `AnomalyDetectors` — isolation forest, local outlier factor, one-class models.
- `Classifiers` — GBDT, random forest, logistic regression, SVM, KNN, naive Bayes, ensembles.
- `Regression` — GBDT regressor, linear models, KNN regressor, tree regressors.
- `Clusterers` — KMeans, DBSCAN, Gaussian mixtures.
- `Decomposition` and `Mainfold` — PCA, t-SNE.
- `Meta` — grid search and random search wrappers.
- `Trees` — k-d trees and decision tree utilities.

## Transformer categories

The `src/Transformers/` directory contains preprocessing operators such as:

- feature scaling: `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `L1Normalizer`, `L2Normalizer`
- encoding: `OneHotLabelEncoder`, `OrdinalEncoder`, `CategoricalEncoder`, `TargetEncoder`
- imputation: `Imputer`, `KNNImputer`
- text: `TextNormalizer`, `TfIdfTransformer`, `WordCountVectorizer`, `TokenHashingVectorizer`
- image: `ImageVectorizer`, `ImageResizer`, `ImageRotator`
- data balancing: `SMOTE`, `TomekLinks`, `NeighborhoodClearing`

## Tokenizer categories

- `Pml\Tokenizers\Tokenizer` — base interface.
- `Whitespace`, `Word`, `WordStemmer`, `Sentence`, `NGram`, `CharGram`, `KSkipNGram`.

## How to read this reference

- `datasets.md` explains data ingestion and ETL.
- `tensor.md` explains the numeric backend and FFI behaviors.
- `models.md` explains estimator interfaces and example models.
- `training.md` explains the trainer, arguments, and callback hooks.
- `nlp.md` explains tokenization and text transformation.
- `vision.md` explains image transformer primitives.
- `ffi.md` explains the C binding layer and memory safety rules.
