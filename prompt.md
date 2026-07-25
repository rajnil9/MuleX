# SYSTEM INSTRUCTION & INFERENCE SPECIFICATION FILE: BANK ACCOUNT FRAUD DETECTION

## 1. ROLE & SYSTEM CONTEXT
Act as a Principal AI/ML Engineer with 20+ years of experience in Enterprise Anti-Money Laundering (AML) and Financial Fraud Prevention.

The objective is to build, evaluate, and deploy an enterprise-grade machine learning system to detect mule account creation and account opening fraud using the NeurIPS 2022 Bank Account Fraud (`Base.csv`) dataset.

---

## 2. DATASET & SPLITTING SPECIFICATIONS
- Dataset File: `data/Base.csv` (~1,000,000 application records across 8 months).
- Target Label: `fraud_bool` (0 = Legitimate, 1 = Fraudulent).
- Imbalance Strategy: Cost-sensitive loss weighting (`scale_pos_weight ~= 90`) to account for the ~1.1% ground-truth fraud prevalence.
- Missing Value Handling: Convert `-1` sentinel values in columns like `prev_address_months_count`, `current_address_months_count`, `intended_balcon_amount`, and `bank_branch_count` to `np.nan` prior to scaling/imputation.
- Temporal Train-Validation-Test Partitioning (To prevent future data leakage):
  * Training Set: Months 0 to 5 (~750,000 rows)
  * Validation Set: Month 6 (~125,000 rows for hyperparameter tuning & early stopping)
  * Out-of-Time Test Set: Month 7 (~125,000 rows for final model benchmarking)

---

## 3. MODEL ARCHITECTURE & EVALUATION
- Primary Model: Stacking Meta-Learner Ensemble (LogisticRegression Meta-Learner over LightGBM & XGBoost base models).
- Advanced ML Optimizations:
  * Hyperparameter Tuning: Optuna for base model optimization.
  * Loss Function: Custom Focal Loss to handle severe class imbalance.
  * Feature Engineering: Weight of Evidence (WOE) Encoding and non-linear Interaction Ratios.
- Decision Thresholding:
  * Low Risk (Score < 20): Approved 🟢
  * Medium Risk (Score 20–70): Step-Up Verification / OTP Required 🟡
  * High Risk (Score > 70): Freeze Application & Flag for AML Operations Queue 🔴
- Core Metrics: Precision-Recall AUC (PR-AUC), ROC-AUC, and Recall @ 5% False Positive Rate (FPR).

---

## 4. SAMPLE TERMINAL INFERENCE OUTPUT FORMATS

### Case 1: High Risk Payload (Mule Account Creation / Synthetic Identity)
```text
============================================================
MULE ACCOUNT DETECTION - INFERENCE RESULTS
============================================================
Application ID : APP-8f92a1c4-2026
Risk Tier      : High Risk (FROZEN) 🔴
Risk Score     : 94.8 / 100.0
------------------------------------------------------------

[+] Top Risk-Increasing Factors (Red Flags):
  - Severe risk due to high device email velocity ('device_distinct_emails_8w' = 15).
  - Severe risk due to prior fraud history on device ('device_fraud_count' = 4).
  - Severe risk due to extreme credit request ($5,000.0) relative to low income (0.05).
  - Severe risk due to complete name/email dissimilarity ('name_email_similarity' = 0.01).
  - Severe risk due to missing address and bank history (-1 sentinel missingness).

[-] Top Trust-Increasing Factors (Green Flags):
  - None identified.

[i] Application Summary & Decision Narrative:
  - Income: 0.05
  - Requested Credit Limit: $5,000.0
  - Device Email Count (8w): 15
  - Action Taken: Application frozen automatically. Flagged for immediate AML operations 
    review due to device farming patterns and severe credit-to-income mismatch.
============================================================


============================================================
MULE ACCOUNT DETECTION - INFERENCE RESULTS
============================================================
Application ID : APP-3b41e8d9-2026
Risk Tier      : Approved 🟢
Risk Score     : 3.2 / 100.0
------------------------------------------------------------

[+] Top Risk-Increasing Factors (Red Flags):
  - None identified.

[-] Top Trust-Increasing Factors (Green Flags):
  - Trust-increasing factor: long-standing current address tenure ('current_address_months_count' = 120).
  - Trust-increasing factor: strong match between name and email ('name_email_similarity' = 0.95).
  - Trust-increasing factor: conservative credit limit request ($150.0) relative to income (0.90).
  - Trust-increasing factor: clean device history ('device_distinct_emails_8w' = 1).

[i] Application Summary & Decision Narrative:
  - Income: 0.90
  - Requested Credit Limit: $150.0
  - Device Email Count (8w): 1
  - Action Taken: Automatically approved. High identity stability and long address history 
    confirm a low-risk, legitimate customer application.
============================================================


============================================================
MULE ACCOUNT DETECTION - INFERENCE RESULTS
============================================================
Application ID : APP-7e10c5a2-2026
Risk Tier      : Medium Risk (MFA / OTP REQUIRED) 🟡
Risk Score     : 51.4 / 100.0
------------------------------------------------------------

[+] Top Risk-Increasing Factors (Red Flags):
  - Moderate flag due to requested credit limit ($800.0) relative to mid-tier income (0.35).
  - Moderate flag due to short address history ('current_address_months_count' = 11).
  - Moderate flag due to multiple email addresses tied to device ('device_distinct_emails_8w' = 3).

[-] Top Trust-Increasing Factors (Green Flags):
  - Trust-increasing factor: verified mobile and home phone numbers.
  - Trust-increasing factor: zero prior fraud records on device ('device_fraud_count' = 0).

[i] Application Summary & Decision Narrative:
  - Income: 0.35
  - Requested Credit Limit: $800.0
  - Device Email Count (8w): 3
  - Action Taken: Step-Up Verification triggered. Application requires multi-factor authentication 
    (MFA/OTP) or document re-verification before account activation.
============================================================