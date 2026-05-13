# PML Examples

Production-grade ML examples demonstrating real business value.
Every script generates its own synthetic data — run any file directly:

```bash
php examples/classification/fraud_detection.php
```

---

## Structure

| Domain | Script | Model | Use Case |
|--------|--------|-------|----------|
| **classification** | `fraud_detection.php` | GBDTClassifier | Credit card fraud |
| | `customer_churn.php` | RandomForestClassifier | Telecom churn |
| | `loan_default.php` | GBDT + Pipeline | Loan underwriting |
| **regression** | `house_price.php` | GBDTRegressor | Real estate AVM |
| | `demand_forecasting.php` | GBDTRegressor | Retail demand |
| **clustering** | `customer_segmentation.php` | KMeans + RFM | Marketing segments |
| **anomaly_detection** | `server_intrusion.php` | IsolationForest | Network security |
| | `iot_sensor_anomaly.php` | IsoForest + RobustZScore | Predictive maintenance |
| **nlp** | `sentiment_analysis.php` | TF-IDF + GBDT | Review scoring |
| | `ticket_classification.php` | TF-IDF + RandomForest | Support routing |
| **neural_networks** | `tabular_insurance.php` | Sequential MLP + AdamW | Claims approval |
| **time_series** | `crypto_forecasting.php` | GBDT + technicals | BTC direction |
| **quantitative_finance** | `alpha_factor_model.php` | GBDT + factors | Equity alpha |
| **cybersecurity** | `malware_classification.php` | RandomForest | PE file scanning |
| **healthcare** | `diabetes_prediction.php` | GBDT + Pipeline | Clinical risk |
| **recommendation** | `item_similarity.php` | PCA + KMeans + cosine | E-commerce recs |
| **dimensionality_reduction** | `customer_embedding.php` | PCA | Behaviour embedding |
| **feature_engineering** | `full_pipeline.php` | Pipeline + GBDT | Attrition prediction |
| **online_learning** | `streaming_fraud.php` | MLP + partial() | Concept drift |
| **tensor_engine** | `tensor_ops_benchmark.php` | Tensor | AVX2/BLAS benchmark |
| **real_world_apps** | `fraud_scoring_api.php` | Pipeline | Production API pattern |
| | `log_anomaly_detector.php` | IsoForest + RobustZ | AIOps / SRE |
| | `mini_bloomberg_ai.php` | KMeans+GBDT+IsoForest | Quant terminal |
| **vision** | `image_classification.php` | CNN (Conv2D + GAP) | Medical X-Ray screening |
| | `object_detection.php` | HOG + GBDT + NMS | Manufacturing defect detection |
| | `semantic_segmentation.php` | Patch-MLP (per-pixel) | Satellite land-use mapping |
| | `video_motion_detection.php` | IsolationForest | Retail loss prevention (CCTV) |
| | `video_action_recognition.php` | Temporal MLP + BN | Workplace safety monitoring |
| | `image_generation.php` | Conditional MLP Generator | Text-to-image synthesis |
| | `super_resolution.php` | Residual SRCNN + resize utils | 4× medical scan upscaling |

---

## Quick Start

```bash
# Any single example
php examples/classification/fraud_detection.php

# Run all
find examples -name "*.php" ! -name "bootstrap.php" | sort | xargs -I{} php {}
```

---

## Requirements

- PHP 8.1+
- `src/Lib/libtensor.so` compiled (see CLAUDE.md)
- `composer install`
