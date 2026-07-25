import logging
import argparse
import pandas as pd
import numpy as np
from src.pipeline import DataLoader, FraudFeatureEngineer, DataSplitter, ImbalanceHandler
from src.models import EnsembleModel, RiskEngine, SHAPExplainer, ModelTuner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MuleFraudPipeline:
    def __init__(self, data_path: str):
        self.data_path = data_path
        
    def run(self):
        logging.info(f"Starting Mule Fraud Pipeline for dataset: {self.data_path}")
        
        # 1. Ingestion & Hygiene
        loader = DataLoader(data_path=self.data_path)
        try:
            raw_data = loader.load_data()
            df = loader.preprocess_missing_values(raw_data)
        except FileNotFoundError:
            logging.error(f"Dataset not found at {self.data_path}. Aborting pipeline.")
            return
            
        # 2. Split (Temporal Walk-Forward)
        splitter = DataSplitter()
        train_df, val_df, test_df = splitter.split(df)
        
        # Combine train and val for Walk-Forward tuning (Months 0-6)
        tune_df = pd.concat([train_df, val_df]).reset_index(drop=True)
        cv_splits = splitter.get_cv_splits(tune_df)
        
        target_col = 'fraud_bool'
        drop_cols = [target_col, 'month', 'application_id']
        
        y_tune = tune_df[target_col]
        X_tune = tune_df.drop(columns=[c for c in drop_cols if c in tune_df.columns])
        
        y_test = test_df[target_col]
        X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
        
        # 3. Feature Engineering
        engineer = FraudFeatureEngineer(use_target_encoding=True)
        X_tune_eng = engineer.process_all(X_tune, target=y_tune, is_train=True)
        X_test_eng = engineer.process_all(X_test, is_train=False)
        
        # 4. Imbalance Handling
        imb_handler = ImbalanceHandler()
        scale_weight = imb_handler.calculate_scale_pos_weight(y_tune)
        
        # 5. Optuna Tuning (10 trials to save time as noted in plan)
        tuner = ModelTuner(X_tune_eng, y_tune, cv_splits)
        best_lgb_params = tuner.tune(n_trials=10)
        
        # 6. Meta-Learner Stacking Training
        ensemble = EnsembleModel(lgb_params=best_lgb_params, scale_pos_weight=scale_weight)
        ensemble.fit(X_tune_eng, y_tune, cv_splits=cv_splits)
        
        # 7. Calibration on final fold (Month 6) for robust thresholds
        logging.info("Fitting Isotonic Calibration on Validation Set (Month 6)...")
        val_idx = cv_splits[-1][1] # Validation indices of the last fold (Month 6)
        y_val_calib = y_tune.iloc[val_idx]
        X_val_calib_eng = X_tune_eng.iloc[val_idx]
        
        val_probs = ensemble.predict_proba(X_val_calib_eng)[:, 1]
        
        from sklearn.isotonic import IsotonicRegression
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(val_probs, y_val_calib)
        
        # Predict & Calibrate on Test set
        test_probs_raw = ensemble.predict_proba(X_test_eng)[:, 1]
        test_probs = calibrator.predict(test_probs_raw)
        
        risk_engine = RiskEngine()
        metrics = risk_engine.evaluate(y_test, test_probs)
        results_df = risk_engine.assign_risk(test_probs, index=X_test_eng.index)
        
        # 8. Explainability
        logging.info("Initializing Explainer Engine...")
        explainer = SHAPExplainer(ensemble.lgb_model)
        # SHAP explainer fit on a random subset for speed
        explainer.fit(X_tune_eng.sample(n=min(10000, len(X_tune_eng)), random_state=42))
            
        # 9. Model Persistence
        import joblib
        import os
        os.makedirs("models", exist_ok=True)
        joblib.dump(engineer, "models/engineer.joblib")
        joblib.dump(ensemble, "models/ensemble.joblib")
        joblib.dump(calibrator, "models/calibrator.joblib")
        joblib.dump(explainer, "models/explainer.joblib")
        logging.info("Saved engineer, ensemble, calibrator, and explainer to models/ directory.")
            
        logging.info("Pipeline Complete!")
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Mule Fraud Detection Pipeline')
    parser.add_argument('--data', type=str, default='data/Base.csv', help='Path to dataset')
    args = parser.parse_args()
    
    pipeline = MuleFraudPipeline(args.data)
    pipeline.run()
