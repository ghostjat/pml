<?php
declare(strict_types=1);

namespace Pml;

// Library version
const VERSION = '1.0.0';

// Epsilon values for numerical stability
const EPSILON     = 1e-8;
const EPSILON_F32 = 1.175494e-38;   // smallest positive float32

// Default random seed sentinel (means: use current time)
const NO_SEED = -1;

// Estimator type constants (mirrors EstimatorType class for array contexts)
const CLASSIFIER       = 0;
const REGRESSOR        = 1;
const CLUSTERER        = 2;
const ANOMALY_DETECTOR = 3;
const EMBEDDER         = 4;

// Data type constants (mirrors DataType class)
const CONTINUOUS  = 0;
const CATEGORICAL = 1;
const IMAGE       = 2;
