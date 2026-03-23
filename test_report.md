# Pml Integration Test Report

> Generated: 2026-03-23 06:04:00 | PHP 8.4.16


## Suite: Classic ML Crucible

- ✅ **Pipeline StandardScaler→PCA(2)→RandomForest: accuracy > 90%**
- ✅ **cross_val_score LogisticRegression KFold(5): all scores valid**
- ✅ **Ridge(alpha=0.1) on y=3x₁−2x₂+5+ε: R²>0.85 & coef≈[3,−2]**
- ✅ **Lasso(alpha=0.01) on y=3x₁−2x₂+5+ε: R²>0.85 & coef≈[3,−2]**

> **Memory delta:** PHP heap +10.0 MiB, RSS +18.1 MiB

## Suite: Deep Learning Autograd Engine

- ✅ **MLPClassifier(hidden=[32], 20 epochs): loss decreases end-to-end**
- ✅ **Low-level autograd: 10 AdamW steps decrease cross-entropy loss**
- ✅ **Single backward step: parameter gradient is non-zero**

> **Memory delta:** PHP heap +2.0 MiB, RSS +2.1 MiB

## Suite: Serialization Integrity (Joblib)

- ✅ **GaussianNB: Joblib dump→load predictions match bit-for-bit**
- ✅ **Joblib file is valid PHP-serialized surrogate (no CData)**
- ✅ **GaussianNB predictions tensor shape preserved after Joblib roundtrip**

> **Memory delta:** PHP heap +0.0 MiB, RSS +0.2 MiB

## Suite: Preprocessing & Model Selection

- ✅ **SimpleImputer(mean): zero NaN values remain after fit_transform**
- ✅ **VarianceThreshold(0.0): constant column removed → n_features−1 remain**
- ✅ **OneHotEncoder: [n,1] with {0,1,2} → [n,3], all 0/1, rows sum to 1**
- ✅ **GridSearchCV(Ridge, alpha=[0.01,0.1,1.0]): best_params_ populated, best_score_ finite**

> **Memory delta:** PHP heap +0.0 MiB, RSS +0.2 MiB

## Suite: Clustering & Distance-Based Models

- ✅ **KMeans(k=3) on make_blobs(centers=3): shape, inertia, label coverage**
- ✅ **KNeighborsClassifier(k=5) on Iris: accuracy > 90%**

> **Memory delta:** PHP heap +0.0 MiB, RSS +0.1 MiB

## Suite: Advanced Ensembles & Regularised Linear Models

- ✅ **ElasticNet(α=0.01, l1_ratio=0.5) on synthetic regression: R² > 0.85**
- ✅ **AdaBoostClassifier(50 rounds) on binary make_classification: accuracy > 85%**
- ✅ **BaggingClassifier(DecisionTree base, 20 estimators) on Iris: accuracy > 90%**

> **Memory delta:** PHP heap +0.0 MiB, RSS +0.2 MiB

## Suite: Phase 6: Native C-Bindings & DBSCAN

- ❌ **SVC(RBF, C=10) on Iris: n_classes_=3, gamma_ finite, accuracy > 90%**: RuntimeException: assertGreaterThan failed (SVC accuracy=0.4333 > 0.90): 0.433333 ≤ 0.900000
- ✅ **SVR(RBF, C=1, ε=0.1) on synthetic regression: n_features_in_=2, R² > 0.85**
- ✅ **XGBClassifier(n_estimators=20, max_depth=3) on Iris: objective_=multi:softprob, accuracy > 90%**
- ✅ **DBSCAN(eps=2.0, min_samples=5) on make_blobs(centers=3): ≥3 clusters, noise < 10%**

> **Memory delta:** PHP heap +0.0 MiB, RSS +6.5 MiB

---

## Summary

| Metric | Value |
|--------|-------|
| Status | ❌ FAIL |
| Passed | 22 / 23 |
| Failed | 1 |
| Skipped | 0 |
| Total time | 486 ms |
| Peak PHP RAM | 16 MiB |

### Per-test results

| Suite | Test | Status | Time (ms) | PHP Δ (KB) |
|-------|------|--------|-----------|------------|
| Classic ML Crucible | ✅ Pipeline StandardScaler→PCA(2)→RandomForest: accuracy > 90% | PASS | 88 | +0 |
| Classic ML Crucible | ✅ cross_val_score LogisticRegression KFold(5): all scores valid | PASS | 212 | +10240 |
| Classic ML Crucible | ✅ Ridge(alpha=0.1) on y=3x₁−2x₂+5+ε: R²>0.85 & coef≈[3,−2] | PASS | 2 | +0 |
| Classic ML Crucible | ✅ Lasso(alpha=0.01) on y=3x₁−2x₂+5+ε: R²>0.85 & coef≈[3,−2] | PASS | 2 | +0 |
| Deep Learning Autograd Engine | ✅ MLPClassifier(hidden=[32], 20 epochs): loss decreases end-to-end | PASS | 23 | +2048 |
| Deep Learning Autograd Engine | ✅ Low-level autograd: 10 AdamW steps decrease cross-entropy loss | PASS | 2 | +0 |
| Deep Learning Autograd Engine | ✅ Single backward step: parameter gradient is non-zero | PASS | 0 | +0 |
| Serialization Integrity (Joblib) | ✅ GaussianNB: Joblib dump→load predictions match bit-for-bit | PASS | 7 | +0 |
| Serialization Integrity (Joblib) | ✅ Joblib file is valid PHP-serialized surrogate (no CData) | PASS | 0 | +0 |
| Serialization Integrity (Joblib) | ✅ GaussianNB predictions tensor shape preserved after Joblib roundtrip | PASS | 2 | +0 |
| Preprocessing & Model Selection | ✅ SimpleImputer(mean): zero NaN values remain after fit_transform | PASS | 2 | +0 |
| Preprocessing & Model Selection | ✅ VarianceThreshold(0.0): constant column removed → n_features−1 remain | PASS | 1 | +0 |
| Preprocessing & Model Selection | ✅ OneHotEncoder: [n,1] with {0,1,2} → [n,3], all 0/1, rows sum to 1 | PASS | 0 | +0 |
| Preprocessing & Model Selection | ✅ GridSearchCV(Ridge, alpha=[0.01,0.1,1.0]): best_params_ populated, best_score_ finite | PASS | 6 | +0 |
| Clustering & Distance-Based Models | ✅ KMeans(k=3) on make_blobs(centers=3): shape, inertia, label coverage | PASS | 14 | +0 |
| Clustering & Distance-Based Models | ✅ KNeighborsClassifier(k=5) on Iris: accuracy > 90% | PASS | 3 | +0 |
| Advanced Ensembles & Regularised Linear Models | ✅ ElasticNet(α=0.01, l1_ratio=0.5) on synthetic regression: R² > 0.85 | PASS | 2 | +0 |
| Advanced Ensembles & Regularised Linear Models | ✅ AdaBoostClassifier(50 rounds) on binary make_classification: accuracy > 85% | PASS | 5 | +0 |
| Advanced Ensembles & Regularised Linear Models | ✅ BaggingClassifier(DecisionTree base, 20 estimators) on Iris: accuracy > 90% | PASS | 21 | +0 |
| Phase 6: Native C-Bindings & DBSCAN | ❌ SVC(RBF, C=10) on Iris: n_classes_=3, gamma_ finite, accuracy > 90% | FAIL | 3 | +0 |
| Phase 6: Native C-Bindings & DBSCAN | ✅ SVR(RBF, C=1, ε=0.1) on synthetic regression: n_features_in_=2, R² > 0.85 | PASS | 14 | +0 |
| Phase 6: Native C-Bindings & DBSCAN | ✅ XGBClassifier(n_estimators=20, max_depth=3) on Iris: objective_=multi:softprob, accuracy > 90% | PASS | 28 | +0 |
| Phase 6: Native C-Bindings & DBSCAN | ✅ DBSCAN(eps=2.0, min_samples=5) on make_blobs(centers=3): ≥3 clusters, noise < 10% | PASS | 14 | +0 |
