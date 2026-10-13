# Production Monitoring & Drift Detection Plan

> **DISCLAIMER**: **ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY**  
> Educational prototype for MLOps coursework. Not for direct patient care or clinical diagnosis.

---

## 1. Monitoring Scope & Objectives

In clinical decision-support systems, model performance is susceptible to population health shifts, clinical practice guideline changes, and data pipeline regressions. The monitoring architecture tracks three core pillars:
1. **Input Data Drift**: Shifts in clinical feature distributions between baseline training data and production serving requests.
2. **Serving Data Quality**: Schema compliance, unexpected missing values, and out-of-range clinical measurements.
3. **Algorithmic Fairness & Disparity**: Demographic equity across Age, Gender, and Race.

---

## 2. Data Drift Metrics & Thresholds

Data drift is monitored across 9 key clinical features using **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) two-sample hypothesis testing**.

### Threshold Matrix
| Metric | Stable (Normal) | Warning (Elevated) | Critical (Action Required) | Action Required |
| :--- | :---: | :---: | :---: | :--- |
| **Numerical PSI** | $< 0.10$ | $0.10 \le \text{PSI} < 0.25$ | $\ge 0.25$ | Trigger automated retraining pipeline |
| **Categorical PSI** | $< 0.10$ | $0.10 \le \text{PSI} < 0.25$ | $\ge 0.25$ | Re-fit OneHotEncoder and retrain |
| **KS Test ($p$-value)**| $\ge 0.05$ | $0.01 \le p < 0.05$ | $< 0.01$ | Confirm non-parametric shape shift |
| **Missing Rate** | $< 5\%$ | $5\% - 15\%$ | $> 15\%$ | Block request / Fallback imputer |

### Monitored Feature Registry
- **Continuous Features**: `num_medications`, `number_inpatient`, `time_in_hospital`, `num_lab_procedures`, `number_diagnoses`
- **Categorical & Demographic Features**: `age`, `gender`, `race`, `diag_1_category`

---

## 3. Algorithmic Fairness Monitoring Plan

### Disparity Metrics
1. **Selection Rate**: Proportion of encounters flagged as high risk ($\hat{y} = 1$).
2. **Disparate Impact Ratio (DIR)**:
   $$\text{DIR} = \frac{\text{Selection Rate}_{\text{unprivileged}}}{\text{Selection Rate}_{\text{privileged}}}$$
   - **Target**: Within the EEOC Four-Fifths boundary ($0.80 \le \text{DIR} \le 1.25$).
3. **Equal Opportunity Difference (EOD)**:
   $$\text{EOD} = \text{Recall}_{\text{subgroup}} - \text{Recall}_{\text{reference}}$$
   - **Target**: $|\text{EOD}| \le 0.10$.

### Demographic Baselines (Observed on Test Partition)
- **Gender**: Female ($62.8\%$ sensitivity) vs. Male ($59.9\%$ sensitivity) $\rightarrow \text{DIR} = 1.065$ (**Balanced**).
- **Age**: Geriatric patients ($70+$) exhibit higher selection rate ($56.7\%$) and FPR ($55.5\%$) due to clinical comorbidity and polypharmacy. High sensitivity ($66.1\%$) ensures older patients receive transitional care.
- **Race**: Caucasian ($62.6\%$), African American ($58.6\%$), Hispanic ($81.3\%$). Sensitivity is maintained $\ge 58\%$ across all monitored cohorts.

---

## 4. Serving Latency & Reliability Monitoring

- **Service Level Agreement (SLA)**:
  - Median latency (p50): $< 100$ ms
  - 95th percentile (p95): $< 350$ ms
  - 99th percentile (p99): $< 500$ ms
- **Audit Logging**: Every inference request is recorded in JSON Lines format to `logs/serving_requests.log` with:
  - `timestamp`
  - `request_id`
  - `latency_ms`
  - `predicted_class`
  - `readmission_probability`
  - `top_contributing_features` (SHAP values)
  - `client_ip`

---

## 5. Alert Escalation Matrix

```
Level 1: INFO     -> All PSI < 0.10, latency p95 < 200ms
Level 2: WARNING  -> 1-2 features with PSI in [0.10, 0.25), p95 latency > 350ms
                     Action: Log warning in daily health report; review during weekly sprint.
Level 3: CRITICAL -> Any feature PSI >= 0.25, or test Recall < 0.50, or error rate > 1%
                     Action: Automated retraining triggered (`src/training/retrain.py`);
                             notification logged to engineering dashboard;
                             if candidate fails quality gates, escalate to human engineer.
```
