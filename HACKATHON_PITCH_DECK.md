# HACKATHON PITCH DECK

## 1. TITLE & EXECUTIVE SUMMARY

**Project Name:** MuleX — Enterprise Mule Account & Fraud Detection Engine

**One-Liner Pitch:** "An enterprise-grade, real-time machine learning system detecting synthetic identity and mule account creation at account opening using calibrated ensemble modeling and automated IDP feature engineering."

**Key Highlights:** 
- Sub-10ms inference time 
- Automated PDF/Bank Statement feature extraction 
- Calibrated probability scoring across 3 actionable risk tiers

---

## 2. THE PROBLEM & THE SOLUTION

### The Problem
Financial institutions lose billions annually to mule account networks and synthetic identity fraud during digital account onboarding. Traditional rules fail against fast, low-velocity device farming and missing data sentinels.

### The Solution
A dual-engine pipeline that combines:
1. **Intelligent Document Processing (IDP):** Translates raw uploaded bank statement PDFs into normalized ML features.
2. **Calibrated Ensemble Model:** LightGBM + XGBoost Stacking Ensemble with cost-sensitive loss weighting (`scale_pos_weight ~= 90`) to combat extreme ~1.1% class imbalance.

---

## 3. TECHNICAL ARCHITECTURE & FEATURE MATRIX

**Dataset:** NeurIPS 2022 Bank Account Fraud (`Base.csv` — 1,000,000 application records across 8 temporal months).

**Temporal Data Splitting Protocol:**
- **Training:** Months 0–5
- **Validation:** Month 6 (Hyperparameter tuning & probability calibration)
- **Out-of-Time Test:** Month 7 (Strict temporal leakage prevention)

**30 Core Model Features Summary:**
- *Identity & Stability:* `income`, `name_email_similarity`, `current_address_months_count`, `customer_age`, `housing_status`.
- *Device & Behavioral Telemetry:* `device_distinct_emails_8w`, `device_fraud_count`, `session_length_in_minutes`, `device_os`.
- *Velocity & Risk Signals:* `velocity_6h`, `velocity_24h`, `velocity_4w`, `zip_count_4w`, `proposed_credit_limit`.

**Data Engineering Guardrails:**
- Sentinel value translation (`-1` -> `np.nan`) prior to scaling.
- Imbalance-resilient probability calibration via Isotonic Regression.

---

## 4. OPERATIONAL RISK TIERS & INFERENCE EXAMPLES

The final probability score is scaled to a dynamic Risk Score (0-100) and evaluated against calibrated thresholds:

* 🟢 **Approved (< 20.0):** Immediate approval for low-risk, verified applicants.
* 🟡 **Medium Risk (20.0 – 70.0):** Triggers Step-Up Verification (MFA / OTP or ID re-verification).
* 🔴 **High Risk (> 70.0):** Automatic account freeze and AML operations flag.

### Inference Examples

**🟢 Case 01 (Low Risk)**
```text
============================================================
MULE ACCOUNT DETECTION - INFERENCE RESULTS
============================================================
Application ID : APP-LOW-jdhg64ks
Risk Tier      : Approved 🟢
Risk Score     : 8.2 / 100.0
------------------------------------------------------------
[+] Top Risk-Increasing Factors (Red Flags):
  - (None significant)

[-] Top Trust-Increasing Factors (Green Flags):
  - Trust-increasing factor: Zero prior device fraud records.
  - Trust-increasing factor: Long address history (current_address_months_count = 60).
  
[i] Application Summary & Decision Narrative:
  - Action Taken: Automatically approved. High identity stability and long address history confirm a low-risk, legitimate customer application.
============================================================
```

**🟡 Case 03 (Medium Risk)**
```text
============================================================
MULE ACCOUNT DETECTION - INFERENCE RESULTS
============================================================
Application ID : APP-MED-kjhg87fv
Risk Tier      : Medium Risk (MFA / OTP REQUIRED) 🟡
Risk Score     : 45.1 / 100.0
------------------------------------------------------------
[+] Top Risk-Increasing Factors (Red Flags):
  - Moderate flag due to requested credit limit ($800.0) relative to mid-tier income (0.35).
  - Moderate flag due to short address history ('current_address_months_count' = 11).
  - Moderate flag due to multiple email addresses tied to device ('device_distinct_emails_8w' = 3).

[-] Top Trust-Increasing Factors (Green Flags):
  - Trust-increasing factor: verified mobile and home phone numbers.
  
[i] Application Summary & Decision Narrative:
  - Action Taken: Application flagged for step-up verification. Discrepancies in device telemetry require secondary MFA challenge before account activation.
============================================================
```

**🔴 Case 02 (High Risk)**
```text
============================================================
MULE ACCOUNT DETECTION - INFERENCE RESULTS
============================================================
Application ID : APP-HIGH-mnbvcxz4
Risk Tier      : High Risk (ACCOUNT FROZEN) 🔴
Risk Score     : 94.6 / 100.0
------------------------------------------------------------
[+] Top Risk-Increasing Factors (Red Flags):
  - High severity flag due to short address history ('current_address_months_count' = 2).
  - Severe anomaly in credit utilization intent ('intended_balcon_amount' = 120.0).
  - High risk flag due to suspicious velocity ('velocity_6h' = 8500.0).

[-] Top Trust-Increasing Factors (Green Flags):
  - (None significant)
  
[i] Application Summary & Decision Narrative:
  - Action Taken: Application frozen automatically. Flagged for immediate AML operations review due to severe velocity anomalies and short address history.
============================================================
```

---

## 5. JUDGE Q&A DEFENSE BRIEF (PREDICTED HACKATHON QUESTIONS)

**1. Q: How do you handle extreme class imbalance (only 1.1% fraud)?**
*A:* We utilize cost-sensitive objective functions (`scale_pos_weight ~= 90`) combined with Isotonic Probability Calibration on Month 6 validation data, ensuring risk scores reflect true empirical likelihood.

**2. Q: How do you prevent data leakage in time-series transactional data?**
*A:* We use a strict Out-Of-Time (OOT) split. Months 0–5 train the model, Month 6 calibrates hyperparameters, and Month 7 acts as an untouched benchmark test set.

**3. Q: How does a user interact with this if they don't know 30 technical features?**
*A:* End users only fill out 4 simple fields or upload a Bank Statement PDF. Our backend auto-captures browser/device telemetry, calculates velocity metrics via Redis, and uses an LLM/vision parser to extract financial features from the PDF.

**4. Q: How do you deal with missing values or incomplete applicant data?**
*A:* Missing values in `Base.csv` are encoded as `-1`. Our pipeline sanitizes these into true NaNs prior to transformation, allowing tree models to split on missingness natively without data corruption.

**5. Q: What is the inference latency in production?**
*A:* Model inference alone takes <10ms. End-to-end processing with PDF ingestion completes in under 1.5 seconds.

**6. Q: How do you explain model decisions to compliance officers?**
*A:* We integrate tree-based feature attribution (SHAP values) to automatically generate human-readable "Red Flags" and "Green Flags" alongside every risk score.
