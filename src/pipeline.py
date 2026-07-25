import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataLoader:
    def __init__(self, data_path: str = "data/Base.csv"):
        self.data_path = Path(data_path)
        # Columns known to use -1 as a sentinel value for missing data
        self.sentinel_cols = [
            'prev_address_months_count',
            'current_address_months_count',
            'intended_balcon_amount',
            'bank_branch_count_8w',
            'bank_months_count'
        ]

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            logging.error(f"Data file not found: {self.data_path}")
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
        
        logging.info(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        return df

    def preprocess_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Replacing -1 sentinel values and creating missingness indicators...")
        df_processed = df.copy()
        
        # Harmonize column names (test payloads may pass 'bank_branch_count')
        if 'bank_branch_count' in df_processed.columns:
            df_processed = df_processed.rename(columns={'bank_branch_count': 'bank_branch_count_8w'})
        
        for col in self.sentinel_cols:
            if col in df_processed.columns:
                # Create explicit missingness indicator
                df_processed[f'{col}_is_missing'] = (df_processed[col] == -1).astype(int)
                
        # Globally replace -1 sentinel values with NaN before feature scaling
        df_processed = df_processed.replace(-1, np.nan)
                
        logging.info("Preprocessing complete.")
        return df_processed

    def run_initial_eda(self, df: pd.DataFrame):
        logging.info("Running initial Exploratory Data Analysis (EDA)...")
        
        print("\n--- Basic Statistics ---")
        print(df.describe())
        
        print("\n--- Missing Values Profile ---")
        missing_counts = df.isnull().sum()
        print(missing_counts[missing_counts > 0])
        
        if 'fraud_bool' in df.columns:
            print("\n--- Class Imbalance Check ---")
            class_counts = df['fraud_bool'].value_counts()
            class_props = df['fraud_bool'].value_counts(normalize=True)
            for cls, count in class_counts.items():
                print(f"Class {cls}: {count} ({class_props[cls]*100:.2f}%)")
        
        print("\nEDA complete.")


class DataSplitter:
    def __init__(self, time_col: str = 'month'):
        self.time_col = time_col

    def split(self, df: pd.DataFrame):
        logging.info(f"Splitting data based on temporal column: '{self.time_col}'")
        
        if self.time_col not in df.columns:
            logging.error(f"Time column '{self.time_col}' not found in DataFrame.")
            raise ValueError(f"Time column '{self.time_col}' not found.")
            
        # Months 0-5: Training
        train_mask = df[self.time_col] <= 5
        # Month 6: Validation
        val_mask = df[self.time_col] == 6
        # Month 7: Test
        test_mask = df[self.time_col] == 7
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        
        logging.info(f"Train set (Months 0-5): {len(train_df)} rows")
        logging.info(f"Validation set (Month 6): {len(val_df)} rows")
        logging.info(f"Test set (Month 7): {len(test_df)} rows")
        
        return train_df, val_df, test_df

    def get_cv_splits(self, df: pd.DataFrame):
        """
        Returns a list of (train_idx, val_idx) arrays for Walk-Forward CV.
        Month 0-6 are used for CV. Month 7 is holdout.
        Fold 1: Train 0-3, Val 4
        Fold 2: Train 0-4, Val 5
        Fold 3: Train 0-5, Val 6
        """
        if self.time_col not in df.columns:
            raise ValueError(f"Time column '{self.time_col}' not found.")
            
        splits = []
        # df index should be aligned. We'll use integer location (iloc) indices.
        # Ensure df is reset_index if using position based.
        df_reset = df.reset_index(drop=True)
        for val_month in [4, 5, 6]:
            train_mask = df_reset[self.time_col] < val_month
            val_mask = df_reset[self.time_col] == val_month
            
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            splits.append((train_idx, val_idx))
            
        return splits


class FraudFeatureEngineer:
    def __init__(self, use_target_encoding=True):
        self.use_target_encoding = use_target_encoding
        # OHE for low cardinality
        self.ohe_cols = ['payment_type', 'employment_status', 'housing_status']
        # Target encoding for potentially higher cardinality
        self.target_cols = ['device_os']
        
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # Using WOEEncoder instead of basic TargetEncoder
        if self.use_target_encoding:
            # Note: WOEEncoder needs the target during fit
            self.target_encoder = ce.WOEEncoder(cols=self.target_cols, handle_missing='value', handle_unknown='value')
        else:
            self.target_encoder = None
            self.ohe_cols.extend(self.target_cols)
            
    def engineer_risk_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Engineering Risk Interaction Ratios...")
        df_eng = df.copy()
        
        # Credit Utilization / Risk Ratio
        if 'proposed_credit_limit' in df_eng.columns and 'income' in df_eng.columns:
            df_eng['credit_utilization_ratio'] = df_eng['proposed_credit_limit'] / (df_eng['income'] + 0.01)
        
        # Discrepancy markers between customer_age and address stability
        if 'customer_age' in df_eng.columns and 'current_address_months_count' in df_eng.columns:
            # age in months vs address months
            df_eng['address_age_discrepancy'] = (df_eng['customer_age'] * 12) - df_eng['current_address_months_count']
            # Risk marker: Very old person with very short address history
            df_eng['high_age_short_address'] = ((df_eng['customer_age'] > 60) & (df_eng['current_address_months_count'] < 6)).astype(int)
            
        # Non-linear combinations
        if 'credit_risk_score' in df_eng.columns and 'velocity_4w' in df_eng.columns:
            df_eng['risk_score_velocity_interaction'] = df_eng['credit_risk_score'] * np.log1p(df_eng['velocity_4w'])
            
        if 'income' in df_eng.columns and 'velocity_4w' in df_eng.columns:
            df_eng['income_velocity_ratio'] = df_eng['velocity_4w'] / (df_eng['income'] + 0.01)
            
        return df_eng

    def engineer_behavioral_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Engineering Behavioral Velocity Aggregations...")
        df_eng = df.copy()
        
        # Interactions across velocity variables
        if 'velocity_24h' in df_eng.columns and 'velocity_6h' in df_eng.columns:
            df_eng['velocity_acceleration'] = df_eng['velocity_6h'] / (df_eng['velocity_24h'] + 0.01)
            
        # Device email history risk
        if 'velocity_24h' in df_eng.columns and 'device_distinct_emails_8w' in df_eng.columns:
            df_eng['high_velocity_multiple_emails'] = (df_eng['velocity_24h'] * df_eng['device_distinct_emails_8w'])
            
        return df_eng

    def fit_transform_categorical(self, df: pd.DataFrame, target: pd.Series = None) -> pd.DataFrame:
        logging.info("Fitting and transforming Categorical Encodings...")
        df_eng = df.copy()
        
        # Dynamically detect all object / string / category columns
        cat_cols = df_eng.select_dtypes(include=['object', 'string', 'category']).columns.tolist()
        
        # Determine target encoding columns vs OHE columns
        target_cols_present = [c for c in self.target_cols if c in cat_cols] if self.use_target_encoding else []
        ohe_cols_present = [c for c in cat_cols if c not in target_cols_present]
        
        self.ohe_cols_fit = ohe_cols_present
        self.target_cols_fit = target_cols_present
        
        if ohe_cols_present:
            ohe_encoded = self.ohe.fit_transform(df_eng[ohe_cols_present])
            ohe_feature_names = self.ohe.get_feature_names_out(ohe_cols_present)
            ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=df_eng.index)
            df_eng = pd.concat([df_eng.drop(columns=ohe_cols_present), ohe_df], axis=1)
            
        if self.use_target_encoding and target_cols_present and target is not None:
            df_eng[target_cols_present] = self.target_encoder.fit_transform(df_eng[target_cols_present], target)
            
        return df_eng

    def transform_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Transforming Categorical Encodings...")
        df_eng = df.copy()
        
        ohe_cols_present = getattr(self, 'ohe_cols_fit', [c for c in self.ohe_cols if c in df_eng.columns])
        target_cols_present = getattr(self, 'target_cols_fit', [c for c in self.target_cols if c in df_eng.columns])
        
        if ohe_cols_present:
            ohe_encoded = self.ohe.transform(df_eng[ohe_cols_present])
            ohe_feature_names = self.ohe.get_feature_names_out(ohe_cols_present)
            ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=df_eng.index)
            df_eng = pd.concat([df_eng.drop(columns=ohe_cols_present), ohe_df], axis=1)
            
        if self.use_target_encoding and target_cols_present:
            df_eng[target_cols_present] = self.target_encoder.transform(df_eng[target_cols_present])
            
        return df_eng

    def process_all(self, df: pd.DataFrame, target: pd.Series = None, is_train: bool = True) -> pd.DataFrame:
        df_processed = self.engineer_risk_ratios(df)
        df_processed = self.engineer_behavioral_velocity(df_processed)
        
        if is_train:
            df_processed = self.fit_transform_categorical(df_processed, target)
        else:
            df_processed = self.transform_categorical(df_processed)
            
        return df_processed


class ImbalanceHandler:
    def __init__(self, target_col: str = 'fraud_bool'):
        self.target_col = target_col

    def calculate_scale_pos_weight(self, y: pd.Series) -> float:
        """
        Calculate scale_pos_weight for cost-sensitive learning in XGBoost/LightGBM.
        Formula: count(negative examples) / count(positive examples)
        """
        logging.info("Calculating cost-sensitive loss weights...")
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        
        if pos_count == 0:
            logging.warning("No positive examples found! Returning 1.0")
            return 1.0
            
        weight = float(neg_count) / pos_count
        logging.info(f"Calculated scale_pos_weight: {weight:.4f}")
        return weight

    def apply_smote(self, X: pd.DataFrame, y: pd.Series):
        """
        Alternative baseline using SMOTE to handle imbalance.
        """
        logging.info("Applying SMOTE for baseline benchmarking...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        logging.info(f"SMOTE complete. New shape: {X_resampled.shape}")
        return X_resampled, y_resampled
