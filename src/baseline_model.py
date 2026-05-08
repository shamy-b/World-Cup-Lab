import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, classification_report
import joblib
import os
from ingestion import ChronosIngestor
from features import FeatureEngineer
from elo import EloSystem

class BaselineModel:
    """
    Baseline ML Engine for Chronos.
    Uses XGBoost for match outcome prediction.
    """
    
    def __init__(self, model_path="models/baseline_xgb.joblib"):
        self.model_path = model_path
        self.model = None

    def prepare_data(self):
        print("[*] Preparing data for training...")
        # 1. Ingest
        ingestor = ChronosIngestor()
        ingestor.load_all()
        df = ingestor.results
        
        # 2. Features
        fe = FeatureEngineer()
        df = fe.generate_basic_features(df)
        df = fe.generate_rolling_stats(df)
        
        # 3. Elo
        elo = EloSystem()
        df = elo.process_history(df)
        
        # 4. Final Cleanup
        # Map outcome to 0, 1, 2 for XGBoost (0: Away Win, 1: Draw, 2: Home Win)
        # Original outcomes: -1: Away Win, 0: Draw, 1: Home Win
        df['target'] = df['outcome'] + 1
        
        # Select Features
        feature_cols = [
            'elo_home', 'elo_away', 'elo_diff',
            'rolling_win_rate_5_home', 'rolling_win_rate_5_away',
            'rolling_win_rate_10_home', 'rolling_win_rate_10_away',
            'rolling_gf_5_home', 'rolling_ga_5_home',
            'rolling_gf_5_away', 'rolling_ga_5_away'
        ]
        
        # Drop rows with NaNs (mostly early matches where rolling stats aren't available)
        df = df.dropna(subset=feature_cols)
        
        return df, feature_cols

    def train(self, df, feature_cols):
        print("[*] Training Baseline XGBoost model...")
        
        # Chronological Split
        # Train: matches before 2018
        # Test: matches from 2018 onwards
        train_df = df[df['date'] < '2018-01-01']
        test_df = df[df['date'] >= '2018-01-01']
        
        X_train = train_df[feature_cols]
        y_train = train_df['target']
        X_test = test_df[feature_cols]
        y_test = test_df['target']
        
        print(f"    - Train size: {len(X_train)}")
        print(f"    - Test size: {len(X_test)}")
        
        self.model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            objective='multi:softprob',
            random_state=42
        )
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Eval
        preds = self.model.predict(X_test)
        probs = self.model.predict_proba(X_test)
        
        print("\n[+] Baseline Evaluation:")
        print(f"    - Accuracy: {accuracy_score(y_test, preds):.4f}")
        print(f"    - Log Loss: {log_loss(y_test, probs):.4f}")
        print("\nClassification Report:\n", classification_report(y_test, preds, target_names=['Away', 'Draw', 'Home']))
        
        # Save
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(self.model, self.model_path)
        print(f"[*] Model saved to {self.model_path}")

if __name__ == "__main__":
    baseline = BaselineModel()
    data, features = baseline.prepare_data()
    baseline.train(data, features)
