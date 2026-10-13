# MLflow Model Registry and Versioning Architecture
**Academic Decision-Support Prototype Only — Not for Clinical Use**

## 1. Overview
The **MLflow Model Registry** provides centralized model governance, lineage tracking, and environment-agnostic deployment aliases. Rather than serving static filesystem paths or hardcoded run IDs, the serving layer resolves the active model dynamically by **alias** (e.g., `@production`).

---

## 2. Registered Model Metadata

* **Model Name**: `HospitalReadmissionClassifier`
* **Current Version**: `1`
* **Algorithm**: Tuned Gradient Boosted Decision Trees (XGBoost) with Platt Sigmoid Probability Calibration
* **Assigned Aliases**:
  * `@production` (Current live inference endpoint target)
  * `@champion` (Highest overall performance benchmark)
  * `@staging` (Validation testbed target)

### Associated Tags & Provenance
```json
{
  "model_type": "XGBoost",
  "data_version": "v1.0_anti_leakage_grouped",
  "recall_test": "0.6150",
  "precision_test": "0.1497",
  "roc_auc_test": "0.6273",
  "brier_score_test": "0.0973",
  "decision_threshold": "0.10",
  "disclaimer": "Academic decision-support prototype only. Not for clinical diagnostic use."
}
```

---

## 3. Serving Layer Integration (Loading by Alias)

In the serving application (FastAPI), models are loaded using the alias resolution pattern:

```python
from src.serving.registry import load_model_by_alias

# Dynamically loads the model assigned to '@production'
bundle = load_model_by_alias(
    model_name="HospitalReadmissionClassifier",
    alias="production"
)

model = bundle["model"]
calibrator = bundle["calibrator"]
threshold = bundle["threshold"]
preprocessor = bundle["preprocessor"]
version = bundle["version"]
```

### Benefits of Alias-Driven Serving
1. **Zero-Downtime Model Promotion**:
   When a new candidate model passes validation gates, updating the alias:
   ```python
   client.set_registered_model_alias("HospitalReadmissionClassifier", alias="production", version=new_version)
   ```
   promotes the new version immediately without code changes or container redeployments.
2. **Instant Rollback**:
   If monitoring detects anomalous prediction drift or performance degradation, re-pointing `@production` to the previous version takes effect immediately.
