# MuleX: Enterprise Mule Account & Fraud Detection Engine

MuleX is a production-grade machine learning system designed to detect **Mule Accounts** and **Synthetic Identity Fraud** during bank account onboarding. It combines a **Stacking Meta-Learner Ensemble** with **SHAP Explainability** and **Calibrated Risk Tiers** to process customer applications in sub-10 milliseconds.

---

## 📊 Key Model Performance

Evaluated on the NeurIPS 2022 Bank Account Fraud dataset using **Month 7 Out-of-Time (OOT) Testing**:

| Metric | Score | Production Impact |
| :--- | :--- | :--- |
| **Overall Accuracy** | **98.5%** | Correct classification rate across legitimate and fraudulent applications. |
| **ROC-AUC** | **0.896** | High discrimination power ranking fraud above legitimate applications. |
| **Recall @ 5% FPR** | **54.2%** | Catches **over half of all fraud attempts** while keeping customer friction under 5%. |
| **PR-AUC** | **0.280** | Resilient precision-recall trade-off under extreme (~1.1%) class imbalance. |

---

## 🧠 Architecture Overview

MuleX utilizes a multi-layered Stacking Ensemble architecture:

1. **Level 0 Base Models**:
   - **LightGBM Classifier**: Trained using a custom mathematical **Focal Loss** function to aggressively penalize hard-to-detect fraud cases.
   - **XGBoost Classifier**: Deeply regularized gradient boosting model using `scale_pos_weight` (~90) for imbalanced data.
2. **Level 1 Meta-Learner**:
   - **Logistic Regression**: Combines predictions from LightGBM and XGBoost to generate a unified fraud probability score.
3. **Probability Calibration & Interpretability**:
   - **Isotonic Regression Calibrator**: Scales probabilities into exact Risk Scores (0.0 to 100.0).
   - **SHAP TreeExplainer**: Extracts localized feature importance to generate human-readable **Red Flags** and **Green Flags**.

---

## 🚦 Decision Tiers

Applications are scored from `0.0` to `100.0` and assigned to action tiers:
- 🟢 **Approved (< 20.0)**: Low-risk applications automatically approved.
- 🟡 **Medium Risk (20.0 – 70.0)**: Triggers Step-Up Verification (MFA / OTP required).
- 🔴 **High Risk (> 70.0)**: Account frozen automatically; flagged for immediate AML compliance review.

---

## 📁 Repository Structure

```
MuleX/
├── data/
│   └── Base.csv                # NeurIPS 2022 dataset (~1M rows, tracked via Git LFS)
├── models/
│   ├── engineer.joblib         # WOE Encoder & preprocessor pipeline
│   ├── ensemble.joblib         # Trained Stacking Meta-Learner ensemble
│   ├── calibrator.joblib       # Isotonic probability calibrator
│   └── explainer.joblib        # SHAP explainability model
├── src/
│   ├── pipeline.py             # Feature engineering & temporal walk-forward splitter
│   ├── models.py               # Optuna tuning, Focal Loss, & Stacking Ensemble logic
│   ├── main.py                 # Full training & calibration pipeline runner
│   └── inference.py            # CLI inference & SHAP explanation engine
├── batch_test_cases.json       # 20 simulated application test benchmark suite
├── HACKATHON_PITCH_DECK.md     # Master pitch deck & judge Q&A defense brief
├── model_summary.md            # Technical model summary
└── project_structure.md        # Comprehensive file guide
```

---

## 🚀 Quickstart & Inference CLI

### Run a Specific Test Case (`CASE_01` to `CASE_20`)
```powershell
python -m src.inference --case CASE_01
```

### Run by Application ID (`--id`)
```powershell
python -m src.inference --id APP-HIGH-mnbvcxz4
```

### Evaluate All 20 Benchmark Cases Sequentially
```powershell
1..20 | ForEach-Object { $c = "CASE_{0:D2}" -f $_; Write-Host "`n>>> Running $c..." -ForegroundColor Cyan; python -m src.inference --case $c }
```
