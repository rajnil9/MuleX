# MuleX Fraud Detection Model Summary

This document provides a simple and concise overview of the current state of the machine learning pipeline used for detecting Mule Accounts and synthetic identities.

## 1. Overview
The system employs an advanced **Stacking Meta-Learner Ensemble** to detect anomalous and fraudulent applications. It is designed to handle severe class imbalance (~1.1% fraud rate) and prevent data leakage over time.

## 2. Data Preprocessing & Splitting
- **Dataset:** NeurIPS 2022 Bank Account Fraud (`Base.csv`) containing ~1 million records over 8 months.
- **Sentinel Value Handling:** Missing values, initially marked as `-1`, are explicitly converted to `NaN` to prevent models from interpreting them as extreme numerical features. A corresponding "is_missing" binary indicator is created.
- **Temporal Walk-Forward Splitting:** To mimic a real-world production environment and prevent future data leakage:
  - **Training:** Months 0 to 5
  - **Validation:** Month 6 (used for early stopping and hyperparameter tuning)
  - **Testing:** Month 7 (out-of-time benchmarking)

## 3. Advanced Feature Engineering
- **Weight of Evidence (WOE) Encoding:** Categorical variables are transformed using WOE Encoding. This technique calculates the logarithmic odds of fraud for each category, providing a highly predictive, continuous numerical value to the tree algorithms.
- **Interaction Ratios:** The pipeline engineers specialized non-linear mathematical combinations to expose hidden fraud structures, such as:
  - `velocity_4w / credit_risk_score`
  - `requested_credit_limit / income`
- **Behavioral Velocity Aggregations:** Features tracking the frequency of emails and applications tied to a device over 4-week and 8-week periods are aggressively leveraged.

## 4. Model Architecture & Algorithms
The core model is a Stacking Ensemble, which combines the strengths of multiple algorithms:

### Base Models (Level 0)
1. **LightGBM (`lgb.LGBMClassifier`):** A fast, gradient-boosting framework optimized for large datasets.
   - **Custom Focal Loss:** Instead of standard log-loss, LightGBM is trained using a specialized mathematical Focal Loss function. This penalizes the model for misclassifying "hard" fraud cases while reducing the focus on easily predictable legitimate customers.
2. **XGBoost (`xgb.XGBClassifier`):** A highly robust and deeply regularized gradient-boosting tree model, utilizing `scale_pos_weight` (~90) to handle class imbalance.

### Meta-Learner (Level 1)
- **Algorithm:** `sklearn.linear_model.LogisticRegression`
- **How it works:** Instead of simply averaging the scores of LightGBM and XGBoost, the Meta-Learner trains on the out-of-fold predictions of both base models. It intelligently learns which model to trust more in specific feature spaces to produce the final `P(fraud)` probability.

### Optuna Hyperparameter Tuning
Before training the final ensemble, the pipeline dynamically explores and optimizes the hyperparameter space (e.g., `learning_rate`, `max_depth`, `num_leaves`, `colsample_bytree`) for the base models using **Optuna**, ensuring peak performance.

## 5. Decision Thresholds & Risk Tiers
The final probability score is scaled to a dynamic Risk Score (0-100) and evaluated against calibrated thresholds:
- 🟢 **Approved (Score < 20.0):** Standard low-risk flow.
- 🟡 **Medium Risk (Score 20.0 - 70.0):** Step-Up Verification triggered; Multi-Factor Authentication (MFA / OTP) required.
- 🔴 **High Risk (Score > 70.0):** Application frozen automatically and flagged for immediate AML operations review.

## 6. Local Interpretability (SHAP)
During inference, the pipeline leverages **SHAP (SHapley Additive exPlanations)** on the LightGBM branch of the ensemble to extract localized feature importance. This provides human-readable explanations (Red Flags and Green Flags) for exactly *why* a specific application was approved or frozen, ensuring regulatory compliance and operational transparency.

## 7. Model Evaluation Metrics
Evaluated on the Month 7 Out-of-Time (OOT) test set:
- **Overall Accuracy:** 98.5%
- **ROC-AUC Score:** 0.896 (89.6%)
- **Recall @ 5% FPR:** 54.2% (catches over half of all fraud while limiting false positives to <5%)
- **PR-AUC Score:** 0.280 (resilient performance under ~1.1% class imbalance)
