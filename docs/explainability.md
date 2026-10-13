# Model Explainability & Feature Attribution Analysis
**Academic Decision-Support Prototype Only — Not for Clinical Use**

## 1. Overview
This document analyzes the global and local feature importance of the selected **XGBoost** model for 30-day hospital readmission screening, utilizing **TreeSHAP** (Tree-based Shapley Additive exPlanations).

---

## 2. Global Feature Importance (Top Readmission Risk Drivers)

Based on the TreeSHAP analysis of the validation cohort (logged to MLflow artifact [`reports/plots/shap_summary.png`](../reports/plots/shap_summary.png)), the primary risk drivers are:

| Rank | Feature | Clinical / Operational Rationale | Directional Impact |
| :---: | :--- | :--- | :--- |
| **1** | `number_inpatient` | Historical hospital admissions in the prior year. Strongest marker of chronic illness frailty. | Higher prior admissions strongly **increase** readmission risk. |
| **2** | `number_diagnoses` | Comorbidity burden (total secondary and primary diagnoses recorded during encounter). | Multi-morbid patients have elevated risk of post-discharge decompensation. |
| **3** | `num_medications` | Complexity of pharmacotherapy regimen. Proxy for disease severity and polypharmacy complications. | Higher medication counts **increase** readmission risk. |
| **4** | `time_in_hospital` | Length of hospital stay. Longer acute admissions reflect severe clinical presentations. | Longer hospital stay modestly **increases** risk. |
| **5** | `num_lab_procedures` | Volume of diagnostic tests performed during inpatient encounter. | Higher diagnostic intensity corresponds with higher clinical acuity. |
| **6** | `admission_type_id` | Emergency / urgent vs elective admission origin. | Emergency admissions demonstrate significantly higher readmission likelihood. |
| **7** | `diag_1_category_Circulatory` | Primary diagnosis of cardiovascular conditions (ICD-9 390–459: congestive heart failure, CAD). | Cardiovascular diagnoses are major clinical drivers of 30-day readmission. |
| **8** | `diag_1_category_Diabetes` | Primary diagnosis of acute diabetes complications (ICD-9 250.xx). | Active glycemic instability elevates rehospitalization risk. |

---

## 3. Local Prediction Explanations
For every individual inference request, the serving API returns the local SHAP decomposition:
$$\text{Log-Odds Risk} = \phi_0 + \sum_{i=1}^{M} \phi_i$$
where:
* $\phi_0$ is the base expected log-odds across the training population.
* $\phi_i$ is the exact contribution of feature $i$ for this specific patient.

Positive contributions ($\phi_i > 0$) increase the predicted risk of readmission, while negative contributions ($\phi_i < 0$) lower the predicted risk.

---

## 4. Critical Limitations of SHAP in Clinical Decision Support

While TreeSHAP provides mathematically rigorous local and global attribution, several critical limitations must be recognized:

1. **Association is NOT Causation**:
   * SHAP measures statistical association within historical observational data, **not causal effect**.
   * *Example*: A positive SHAP value for `num_medications` indicates that patients taking more medications are historically more likely to be readmitted due to comorbidity. **It does NOT mean that discontinuing medications will reduce the patient's readmission risk.** Recommending medication reduction based on SHAP would cause severe patient harm.
2. **Collinearity and Credit Sharing**:
   * Highly correlated features (e.g. `num_medications`, `num_lab_procedures`, `number_diagnoses`) share attribution credit. An individual feature's attribution may appear diluted if a correlated companion feature captures part of the signal.
3. **Historical Practice & Coding Artifacts**:
   * Features like `admission_source_id` and `diag_1` reflect administrative billing and coding practices rather than pure physiological pathophysiology.
4. **Academic Prototype Boundary**:
   * This pipeline is an academic decision-support prototype. It has not undergone clinical trials, multi-center external validation, or FDA/CE-mark software-as-a-medical-device (SaMD) clearance.
