# Model Card: Hospital Readmission Risk Classifier

> **DISCLAIMER**: **ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY**  
> This model is developed strictly for educational, academic, and research purposes as part of an Individual MLOps Coursework Project. It is **NOT** a certified clinical diagnostic device, medical software (SaMD), or triaging instrument, and must **NOT** be used for direct patient care, clinical decision-making, or discharge denial.

---

## 1. Model Details
- **Model Name**: Hospital Readmission Classifier (`HospitalReadmissionClassifier`)
- **Version**: `2.0.0` (MLflow Model Registry alias: `@production`, `@champion`)
- **Model Type**: Calibrated Gradient Boosted Decision Trees (`XGBoost`) with Platt Sigmoid Probability Calibration
- **Developer**: MLOps Project Team (Individual Project Assignment - Data Engineering & MLOps)
- **Release Date**: October 2026
- **License**: Educational / Academic Research Use Only
- **Primary Frameworks**: Scikit-Learn (1.9+), XGBoost (3.2+), MLflow (3.17+), FastAPI (0.143+), Streamlit (1.40+)

---

## 2. Intended Use & Clinical Scope

### Intended Use
- **Decision Support for Discharge Planning**: Identifies hospitalized diabetic patients at elevated risk of unplanned all-cause 30-day hospital readmission to support clinical care coordination teams.
- **Resource Allocation**: Guides non-clinical post-discharge transitional care workflows, such as:
  - Pharmacist medication reconciliation prior to discharge.
  - Follow-up telephone check-in within 48–72 hours post-discharge.
  - Scheduling rapid primary care follow-up within 7–14 days.
  - Diabetic self-management education and blood glucose monitoring review.

### Out-of-Scope & Prohibited Uses
- **Autonomous Clinical Decisions**: Must never be used as an automated decision-maker.
- **Denial of Care**: Must never be used to withhold medical care, deny admission, or prematurely discharge patients.
- **Diagnostic Substitute**: Does not diagnose complications, disease severity, or mortality risk.
- **Non-Diabetic Adult Inpatient Cohorts**: Validated strictly on diabetic inpatient encounters within acute care facilities.

---

## 3. Data & Preprocessing Protocol

### Dataset Source
- **Data Source**: UCI Machine Learning Repository - *Diabetes 130-US Hospitals (1999–2008)*.
- **Total Raw Encounters**: 101,766 encounters across 71,518 unique patients.

### Data Hygiene & Anti-Leakage Protocols
1. **Mortality & Hospice Exclusion**: Excluded 2,423 encounters where `discharge_disposition_id` indicated expiration or hospice transfer (`ids: [11, 13, 14, 19, 20, 21]`). Patients discharged to hospice or deceased cannot experience unplanned community readmission; including them introduces survival bias and clinical leakage.
2. **Patient-Grouped Partitioning**: Standard random row-level splitting causes data leakage when the same patient appears in both training and test partitions. We enforced a **grouped split by `patient_nbr`** (70% train / 15% validation / 15% test):
   - **Training Set**: 69,665 encounters (50,062 unique patients)
   - **Validation Set**: 14,686 encounters (10,728 unique patients)
   - **Test Set**: 14,992 encounters (10,728 unique patients)
   - **Zero Patient Overlap**: Exactly 0 patients overlap across partitions.
3. **Target Definition**: Binary target `readmitted_30d` set strictly to `1` if readmitted within 30 days (`readmitted == '<30'`), and `0` otherwise (`>30` or `NO`), aligning with the CMS Hospital Readmissions Reduction Program (HRRP) benchmark. Base rate: ~11.3%.

---

## 4. Model Training & Evaluation Metrics

### Algorithm Comparison (Held-Out Test Set: 14,992 Encounters)

| Model Architecture | Test Recall | Test Precision | Specificity | F1-Score | ROC-AUC | Brier Score | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.6014 | 0.1444 | 0.5627 | 0.2329 | 0.6133 | 0.0984 | 0.0084 |
| **Random Forest (Trees)**| 0.5896 | 0.1481 | 0.5841 | 0.2368 | 0.6225 | 0.0978 | 0.0076 |
| **XGBoost (Champion)** | **0.6150** | **0.1497** | **0.5773** | **0.2408** | **0.6273** | **0.0973** | **0.0060** |

### Decision Threshold Optimization
- **Operational Priority**: In a high-risk discharge screening system, **sensitivity (Recall)** is prioritized over precision to avoid missing at-risk patients who could benefit from preventive transitional care.
- **Calibrated Threshold**: Tuned on the validation set to **`0.10`** to achieve $\ge 60\%$ screening sensitivity while maintaining clinical specificity $>55\%$.

### Probability Calibration
- Evaluated via Brier score and Expected Calibration Error (ECE, 10 bins).
- Platt Sigmoid calibration reduced Brier loss to **`0.0973`** and ECE to **`0.0060`**, confirming that predicted probabilities closely match observed empirical event rates.

---

## 5. Model Explainability & Key Drivers

Using exact TreeSHAP attribution (`pred_contribs`), the top clinical risk drivers were identified:
1. **`number_inpatient` (Prior Inpatient Admissions)**: Single strongest positive driver of readmission. A history of $\ge 2$ inpatient visits in the prior year substantially increases log-odds risk.
2. **`number_diagnoses` (Comorbidity Burden)**: Elevated claims diagnostic count correlates with multi-organ complexity.
3. **`num_medications` (Polypharmacy)**: Administration of $>15$ distinct medications elevates post-discharge adverse event risk.
4. **`diag_1_category_Circulatory` (Cardiovascular Primary Diagnosis)**: Encounters admitted for congestive heart failure (ICD-9 `428`) or coronary disease (ICD-9 `414`) experience systematically higher readmission frequency.
5. **`admission_type_id_1` (Emergency Admission)**: Unscheduled emergency admission reflects acute clinical instability.

---

## 6. Algorithmic Fairness & Subgroup Disparity Audit

The model was audited across demographic subgroups on the untouched test partition ($N=14,992$):

### Gender Equity
- **Female ($N=8,126$) vs. Male ($N=6,866$)**:
  - Sensitivity: Female $62.8\%$ vs. Male $59.9\%$.
  - Selection Rate: Female $47.7\%$ vs. Male $44.8\%$.
  - **Disparate Impact Ratio**: **`1.065`** (Meets four-fifths rule: $0.80 - 1.25$).

### Age Disparity
- **Under 50 ($N=2,312$)**: Recall $63.0\%$, Selection Rate $34.3\%$, FPR $30.9\%$.
- **50–69 Years ($N=5,907$)**: Recall $55.1\%$, Selection Rate $39.1\%$, FPR $37.2\%$.
- **70+ Geriatric ($N=6,773$)**: Recall $66.1\%$, Selection Rate $56.7\%$, FPR $55.5\%$.
- *Clinical Rationale*: Geriatric patients experience higher selection rates and false positive rates because older diabetic patients carry higher underlying multimorbidity, frailty, and medication burden. This disparity reflects clinical acuity rather than algorithmic bias.

### Racial Subgroups
- **Caucasian ($N=11,322$)**: Sensitivity $62.6\%$, Selection Rate $47.8\%$.
- **African American ($N=2,724$)**: Sensitivity $58.6\%$, Selection Rate $43.8\%$.
- **Hispanic ($N=283$)**: Sensitivity $81.3\%$, Selection Rate $37.1\%$.
- *Clinical Equity*: Sensitivity remains high across all groups ($\ge 58\%$), preventing systematic exclusion of minority cohorts from discharge support services.

---

## 7. Limitations & Caveats
- **Historical Data**: UCI dataset spans 1999–2008. Clinical treatment pathways, electronic health records, and anti-diabetic medications (e.g. SGLT2 inhibitors, GLP-1 receptor agonists) have evolved significantly since this window.
- **Correlation vs. Causation**: SHAP values indicate associative predictive attribution, not causal clinical mechanisms.
- **Missing Socioeconomic Determinants**: The dataset lacks crucial social determinants of health (SDOH), including housing stability, transportation, and health literacy, which strongly influence readmission outcomes.
