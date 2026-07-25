import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, roc_curve
import optuna
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import shap
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def focal_loss_lgb(y_true, y_pred):
    """
    Robust approximate Focal Loss for LightGBM.
    Returns gradient and hessian.
    """
    gamma = 2.0
    # Sigmoid function for raw margins
    p = 1.0 / (1.0 + np.exp(-y_pred))
    
    grad = p - y_true
    # Apply focal weight
    weight = np.power(1.0 - p, gamma) * y_true + np.power(p, gamma) * (1.0 - y_true)
    
    grad = grad * weight
    hess = p * (1.0 - p) * weight
    
    return grad, hess

def focal_loss_lgb_eval(y_true, y_pred):
    """
    Evaluation metric for focal loss.
    """
    gamma = 2.0
    p = 1.0 / (1.0 + np.exp(-y_pred))
    loss = -y_true * np.power(1 - p, gamma) * np.log(np.maximum(p, 1e-15)) - \
           (1 - y_true) * np.power(p, gamma) * np.log(np.maximum(1 - p, 1e-15))
    return 'focal_loss', np.mean(loss), False


class ModelTuner:
    def __init__(self, X, y, cv_splits):
        self.X = X
        self.y = y
        self.cv_splits = cv_splits
        
    def objective(self, trial):
        lgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'is_unbalance': True,
            'verbosity': -1,
            'random_state': 42
        }
        
        pr_aucs = []
        for train_idx, val_idx in self.cv_splits:
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_val, y_val = self.X.iloc[val_idx], self.y.iloc[val_idx]
            
            # Use Focal Loss objective during tuning
            model = lgb.LGBMClassifier(**lgb_params, objective=focal_loss_lgb)
            model.fit(X_train, y_train)
            
            # Since custom objective outputs raw margins, apply sigmoid manually
            margins = model.predict(X_val, raw_score=True)
            y_probs = 1.0 / (1.0 + np.exp(-margins))
            
            pr_auc = average_precision_score(y_val, y_probs)
            pr_aucs.append(pr_auc)
            
        return np.mean(pr_aucs)
        
    def tune(self, n_trials=10):
        logging.info(f"Starting Optuna Hyperparameter Tuning for {n_trials} trials...")
        # Reduce logging for Optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        logging.info(f"Best Optuna Params: {study.best_params}")
        return study.best_params


class EnsembleModel:
    def __init__(self, lgb_params=None, xgb_params=None, scale_pos_weight=1.0):
        if lgb_params is None:
            lgb_params = {'is_unbalance': True, 'verbosity': -1, 'random_state': 42}
        else:
            lgb_params.update({'is_unbalance': True, 'verbosity': -1, 'random_state': 42})
            
        if xgb_params is None:
            xgb_params = {'scale_pos_weight': scale_pos_weight, 'random_state': 42}
        else:
            xgb_params.update({'scale_pos_weight': scale_pos_weight, 'random_state': 42})
            
        # Initialize base models with Focal Loss
        self.lgb_model = lgb.LGBMClassifier(**lgb_params, objective=focal_loss_lgb)
        # XGBoost custom objective handling is sometimes tricky depending on version, sticking to standard logloss + pos weight for robustness.
        # But we inject focal loss into LGBM.
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        
        self.stack = None

    def fit(self, X_train, y_train, cv_splits=None):
        logging.info("Training Meta-Learner Stacking Ensemble...")
        
        meta_learner = LogisticRegression(class_weight='balanced', max_iter=500)
        
        self.stack = StackingClassifier(
            estimators=[
                ('lgb', self.lgb_model),
                ('xgb', self.xgb_model)
            ],
            final_estimator=meta_learner,
            cv=3,
            n_jobs=1  # Sequential to avoid parallel conflict overhead
        )
        self.stack.fit(X_train, y_train)
        
        # Save fitted base models for SHAP
        self.lgb_model = self.stack.named_estimators_['lgb']
        self.xgb_model = self.stack.named_estimators_['xgb']

    def predict_proba(self, X):
        logging.info("Predicting with Stacking Meta-Learner...")
        return self.stack.predict_proba(X)


class RiskEngine:
    def __init__(self):
        # Boundaries out of 100
        self.low_risk_bound = 20.0
        self.high_risk_bound = 70.0

    def score_to_tier(self, score: float) -> str:
        """
        Maps a 0-100 score to Low/Medium/High Risk tier.
        """
        if score < self.low_risk_bound:
            return "Low Risk"
        elif score <= self.high_risk_bound:
            return "Medium Risk"
        else:
            return "High Risk"

    def assign_risk(self, probabilities: np.ndarray, index=None) -> pd.DataFrame:
        """
        Takes raw model probabilities (0.0 to 1.0) and assigns scores and tiers.
        """
        scores = probabilities * 100
        tiers = [self.score_to_tier(s) for s in scores]
        
        df = pd.DataFrame({
            'fraud_probability': probabilities,
            'risk_score': scores,
            'risk_tier': tiers
        }, index=index)
        return df

    def evaluate(self, y_true: np.ndarray, y_probs: np.ndarray):
        logging.info("Evaluating Model Performance...")
        
        pr_auc = average_precision_score(y_true, y_probs)
        roc_auc = roc_auc_score(y_true, y_probs)
        
        logging.info(f"PR-AUC: {pr_auc:.4f}")
        logging.info(f"ROC-AUC: {roc_auc:.4f}")
        
        # Recall at 1% and 5% FPR
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        
        # Helper to find recall at given FPR
        def get_recall_at_fpr(target_fpr):
            idx = np.where(fpr <= target_fpr)[0][-1]
            return tpr[idx]
            
        recall_1_pct = get_recall_at_fpr(0.01)
        recall_5_pct = get_recall_at_fpr(0.05)
        
        logging.info(f"Recall @ 1% FPR: {recall_1_pct:.4f}")
        logging.info(f"Recall @ 5% FPR: {recall_5_pct:.4f}")
        
        return {
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            'Recall@1%FPR': recall_1_pct,
            'Recall@5%FPR': recall_5_pct
        }


class SHAPExplainer:
    def __init__(self, model):
        self.model = model
        self.explainer = None
        
        # Plain English translation dictionary for features
        self.feature_translation = {
            'credit_utilization_ratio': "high proposed credit limit relative to reported income",
            'address_age_discrepancy': "discrepancy between age and address history",
            'high_age_short_address': "very short address history for a senior applicant",
            'velocity_acceleration': "sudden acceleration in application velocity recently",
            'high_velocity_multiple_emails': "high volume of applications combined with multiple email addresses tied to the same device",
            'velocity_24h': "unusually high number of applications in the last 24 hours",
            'income': "reported income level",
            'proposed_credit_limit': "amount of credit requested",
            'device_distinct_emails_8w': "multiple distinct email addresses linked to the same device recently"
        }

    def fit(self, X):
        logging.info("Fitting SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_instance(self, X_instance, detailed=False):
        if self.explainer is None:
            raise ValueError("SHAP explainer not fitted. Call fit(X) first.")
            
        shap_values = self.explainer.shap_values(X_instance)
        
        if isinstance(shap_values, list):
            shap_vals = shap_values[1][0]
        else:
            shap_vals = shap_values[0]
            
        feature_names = X_instance.columns
        contributions = list(zip(feature_names, shap_vals))
        contributions.sort(key=lambda x: x[1], reverse=True)
        
        top_positive = [c for c in contributions if c[1] > 0][:3]
        
        top_negative = [c for c in contributions if c[1] < 0]
        top_negative.sort(key=lambda x: x[1])
        top_negative = top_negative[:3]
        
        plain_text_reasons = []
        for feature, val in top_positive:
            english_desc = self.feature_translation.get(feature, f"anomalous behavior in '{feature}'")
            sentence = f"Flagged due to {english_desc}."
            plain_text_reasons.append(sentence)
            
        if not plain_text_reasons:
            plain_text_reasons.append("No specific high-risk anomalies detected. Score driven by general profile matching.")
            
        negative_reasons = []
        for feature, val in top_negative:
            english_desc = self.feature_translation.get(feature, f"trusted behavior in '{feature}'")
            sentence = f"Trust-increasing factor: {english_desc}."
            negative_reasons.append(sentence)
            
        if detailed:
            return {
                "positive": plain_text_reasons,
                "negative": negative_reasons
            }
            
        return plain_text_reasons
