import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import accuracy_score, log_loss, classification_report, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import optuna
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from ingestion import ChronosIngestor
from features import FeatureEngineer
from elo import EloSystem

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ChronosPredictor:
    """
    Advanced ML Engine for Chronos.
    Ensemble of XGBoost + LightGBM with Optuna hyperparameter tuning
    and probability calibration.
    """
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        self.xgb_model = None
        self.lgb_model = None
        self.meta_model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        os.makedirs(model_dir, exist_ok=True)

    def prepare_data(self):
        """Full pipeline: ingest -> features -> elo -> feature selection."""
        print("=" * 60)
        print("CHRONOS ENGINE v2 — ADVANCED TRAINING PIPELINE")
        print("=" * 60)
        
        # 1. Ingest
        print("\n[Phase 1] Data Ingestion...")
        ingestor = ChronosIngestor()
        ingestor.load_all()
        df = ingestor.results
        
        # 2. Feature Engineering
        print("\n[Phase 2] Advanced Feature Engineering...")
        fe = FeatureEngineer()
        df = fe.generate_all_features(df)
        
        # 3. Elo
        print("\n[Phase 3] Dynamic Elo System...")
        elo = EloSystem()
        df = elo.process_history(df)
        
        # 4. Derived Elo features
        df['elo_expected_home'] = 1 / (1 + 10 ** ((df['elo_away'] - df['elo_home']) / 400))
        df['elo_expected_away'] = 1 - df['elo_expected_home']
        
        # 5. Target
        df['target'] = df['outcome'] + 1  # 0=Away, 1=Draw, 2=Home
        
        # 6. Feature Selection
        self.feature_cols = [
            # Elo features
            'elo_home', 'elo_away', 'elo_diff',
            'elo_expected_home', 'elo_expected_away',
            # Tournament & venue
            'tournament_prestige', 'is_neutral',
            # Rolling goals (home)
            'roll_gf_3_home', 'roll_ga_3_home',
            'roll_gf_5_home', 'roll_ga_5_home',
            'roll_gf_10_home', 'roll_ga_10_home',
            'roll_gf_20_home', 'roll_ga_20_home',
            # Rolling goals (away)
            'roll_gf_3_away', 'roll_ga_3_away',
            'roll_gf_5_away', 'roll_ga_5_away',
            'roll_gf_10_away', 'roll_ga_10_away',
            'roll_gf_20_away', 'roll_ga_20_away',
            # Win rates
            'roll_win_3_home', 'roll_win_5_home', 'roll_win_10_home', 'roll_win_20_home',
            'roll_win_3_away', 'roll_win_5_away', 'roll_win_10_away', 'roll_win_20_away',
            # Draw rates (KEY for draw prediction)
            'roll_draw_3_home', 'roll_draw_5_home', 'roll_draw_10_home',
            'roll_draw_3_away', 'roll_draw_5_away', 'roll_draw_10_away',
            # Clean sheet rates
            'roll_cs_5_home', 'roll_cs_10_home',
            'roll_cs_5_away', 'roll_cs_10_away',
            # Goal difference volatility
            'roll_gd_std_5_home', 'roll_gd_std_10_home',
            'roll_gd_std_5_away', 'roll_gd_std_10_away',
            # Momentum
            'momentum_5_home', 'momentum_10_home',
            'momentum_5_away', 'momentum_10_away',
            # Rest days
            'days_rest_home', 'days_rest_away',
            # Head-to-head
            'h2h_home_win_rate', 'h2h_away_win_rate', 'h2h_draw_rate', 'h2h_matches',
        ]
        
        # Verify all columns exist
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            print(f"[!] Warning: Missing columns: {missing}")
            self.feature_cols = [c for c in self.feature_cols if c in df.columns]
        
        # Drop NaN rows
        df = df.dropna(subset=self.feature_cols)
        
        print(f"\n[+] Total features: {len(self.feature_cols)}")
        print(f"[+] Total usable matches: {len(df)}")
        
        return df

    def _split_data(self, df):
        """Chronological split: train < 2018, validation 2018-2021, test >= 2022."""
        train = df[df['date'] < '2018-01-01']
        val = df[(df['date'] >= '2018-01-01') & (df['date'] < '2022-01-01')]
        test = df[df['date'] >= '2022-01-01']
        
        print(f"\n[+] Chronological Split:")
        print(f"    Train: {len(train)} matches (up to 2017)")
        print(f"    Val:   {len(val)} matches (2018-2021)")
        print(f"    Test:  {len(test)} matches (2022+)")
        
        return train, val, test

    def _tune_xgboost(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optuna hyperparameter search for XGBoost."""
        print("\n[Phase 4a] Tuning XGBoost with Optuna...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'verbosity': 0,
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            probs = model.predict_proba(X_val)
            return log_loss(y_val, probs)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"    Best Log Loss: {study.best_value:.4f}")
        return study.best_params

    def _tune_lightgbm(self, X_train, y_train, X_val, y_val, n_trials=50):
        """Optuna hyperparameter search for LightGBM."""
        print("\n[Phase 4b] Tuning LightGBM with Optuna...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 300, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'objective': 'multiclass',
                'num_class': 3,
                'random_state': 42,
                'verbosity': -1,
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
            
            probs = model.predict_proba(X_val)
            return log_loss(y_val, probs)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"    Best Log Loss: {study.best_value:.4f}")
        return study.best_params

    def train(self, n_trials=50):
        """Full training pipeline with ensemble."""
        df = self.prepare_data()
        train, val, test = self._split_data(df)
        
        X_train = train[self.feature_cols]
        y_train = train['target']
        X_val = val[self.feature_cols]
        y_val = val['target']
        X_test = test[self.feature_cols]
        y_test = test['target']
        
        # === TUNE & TRAIN XGBOOST ===
        xgb_params = self._tune_xgboost(X_train, y_train, X_val, y_val, n_trials)
        xgb_params['objective'] = 'multi:softprob'
        xgb_params['eval_metric'] = 'mlogloss'
        xgb_params['random_state'] = 42
        xgb_params['verbosity'] = 0
        
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        # === TUNE & TRAIN LIGHTGBM ===
        lgb_params = self._tune_lightgbm(X_train, y_train, X_val, y_val, n_trials)
        lgb_params['objective'] = 'multiclass'
        lgb_params['num_class'] = 3
        lgb_params['random_state'] = 42
        lgb_params['verbosity'] = -1
        
        self.lgb_model = lgb.LGBMClassifier(**lgb_params)
        self.lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        # === STACKING ENSEMBLE ===
        print("\n[Phase 5] Building Stacking Ensemble...")
        
        # Get base model predictions on validation set for meta-learner training
        xgb_val_probs = self.xgb_model.predict_proba(X_val)
        lgb_val_probs = self.lgb_model.predict_proba(X_val)
        meta_X_val = np.hstack([xgb_val_probs, lgb_val_probs])
        
        self.meta_model = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_model.fit(meta_X_val, y_val)
        
        # === EVALUATE ON TEST SET ===
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON HELD-OUT TEST SET (2022+)")
        print("=" * 60)
        
        xgb_test_probs = self.xgb_model.predict_proba(X_test)
        lgb_test_probs = self.lgb_model.predict_proba(X_test)
        meta_X_test = np.hstack([xgb_test_probs, lgb_test_probs])
        
        # Individual model results
        xgb_preds = self.xgb_model.predict(X_test)
        lgb_preds = self.lgb_model.predict(X_test)
        ensemble_probs = self.meta_model.predict_proba(meta_X_test)
        ensemble_preds = self.meta_model.predict(meta_X_test)
        
        # Also try simple average
        avg_probs = (xgb_test_probs + lgb_test_probs) / 2
        avg_preds = np.argmax(avg_probs, axis=1)
        
        print("\n--- XGBoost (Tuned) ---")
        print(f"  Accuracy: {accuracy_score(y_test, xgb_preds):.4f}")
        print(f"  Log Loss: {log_loss(y_test, xgb_test_probs):.4f}")
        print(f"  F1 Macro: {f1_score(y_test, xgb_preds, average='macro'):.4f}")
        
        print("\n--- LightGBM (Tuned) ---")
        print(f"  Accuracy: {accuracy_score(y_test, lgb_preds):.4f}")
        print(f"  Log Loss: {log_loss(y_test, lgb_test_probs):.4f}")
        print(f"  F1 Macro: {f1_score(y_test, lgb_preds, average='macro'):.4f}")
        
        print("\n--- Average Ensemble ---")
        print(f"  Accuracy: {accuracy_score(y_test, avg_preds):.4f}")
        print(f"  Log Loss: {log_loss(y_test, avg_probs):.4f}")
        print(f"  F1 Macro: {f1_score(y_test, avg_preds, average='macro'):.4f}")
        
        print("\n--- Stacking Ensemble (Meta-Learner) ---")
        print(f"  Accuracy: {accuracy_score(y_test, ensemble_preds):.4f}")
        print(f"  Log Loss: {log_loss(y_test, ensemble_probs):.4f}")
        print(f"  F1 Macro: {f1_score(y_test, ensemble_preds, average='macro'):.4f}")
        
        # Pick the best model
        results = {
            'XGBoost': (accuracy_score(y_test, xgb_preds), log_loss(y_test, xgb_test_probs)),
            'LightGBM': (accuracy_score(y_test, lgb_preds), log_loss(y_test, lgb_test_probs)),
            'Average': (accuracy_score(y_test, avg_preds), log_loss(y_test, avg_probs)),
            'Stacking': (accuracy_score(y_test, ensemble_preds), log_loss(y_test, ensemble_probs)),
        }
        best_name = min(results, key=lambda k: results[k][1])  # Best by log loss
        best_acc, best_ll = results[best_name]
        
        print(f"\n[🏆] Best Model: {best_name}")
        print(f"     Accuracy: {best_acc:.4f}  |  Log Loss: {best_ll:.4f}")
        
        # Detailed classification report for the best
        if best_name == 'XGBoost':
            best_preds = xgb_preds
        elif best_name == 'LightGBM':
            best_preds = lgb_preds
        elif best_name == 'Average':
            best_preds = avg_preds
        else:
            best_preds = ensemble_preds
            
        print(f"\nClassification Report ({best_name}):")
        print(classification_report(y_test, best_preds, target_names=['Away', 'Draw', 'Home']))
        
        # === FEATURE IMPORTANCE ===
        print("\nTop 15 Features (XGBoost):")
        importance = pd.Series(
            self.xgb_model.feature_importances_, index=self.feature_cols
        ).sort_values(ascending=False)
        for feat, imp in importance.head(15).items():
            bar = "█" * int(imp * 100)
            print(f"  {feat:30s} {imp:.4f} {bar}")
        
        # === SAVE ALL MODELS ===
        joblib.dump(self.xgb_model, os.path.join(self.model_dir, "xgb_tuned.joblib"))
        joblib.dump(self.lgb_model, os.path.join(self.model_dir, "lgb_tuned.joblib"))
        joblib.dump(self.meta_model, os.path.join(self.model_dir, "meta_ensemble.joblib"))
        joblib.dump(self.feature_cols, os.path.join(self.model_dir, "feature_cols.joblib"))
        print(f"\n[*] All models saved to {self.model_dir}/")

if __name__ == "__main__":
    predictor = ChronosPredictor()
    predictor.train(n_trials=50)
