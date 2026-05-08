"""
Phase 3: ML Match Predictor Training
Engineers features from ELO-enriched data and trains a LightGBM classifier.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
import joblib
import optuna
import warnings

warnings.filterwarnings("ignore")

# Paths
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "dataset" / "processed"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

def engineer_features(df):
    print("  Engineering features for ML ...")
    
    # Sort for chronological rolling features
    df = df.sort_values("date").reset_index(drop=True)
    
    # Target variable: result_code (0: away_win, 1: draw, 2: home_win)
    
    # Rolling features per team
    teams = pd.unique(df[["home_team", "away_team"]].values.ravel())
    
    team_stats = []
    for team in teams:
        # Get all matches for this team
        mask_h = df["home_team"] == team
        mask_a = df["away_team"] == team
        
        team_df = df[mask_h | mask_a].copy()
        
        # Points earned: 3 for win, 1 for draw, 0 for loss
        # Note: We must be careful about home/away
        team_df["team_goals"] = np.where(team_df["home_team"] == team, team_df["home_score"], team_df["away_score"])
        team_df["opp_goals"] = np.where(team_df["home_team"] == team, team_df["away_score"], team_df["home_score"])
        
        team_df["team_result"] = np.where(team_df["team_goals"] > team_df["opp_goals"], 3,
                                          np.where(team_df["team_goals"] == team_df["opp_goals"], 1, 0))
        
        # Rolling PPG, goals scored, goals conceded (last 5, 10)
        # Shift(1) to avoid leakage
        team_df["rolling_ppg_5"] = team_df["team_result"].shift(1).rolling(5).mean()
        team_df["rolling_goals_5"] = team_df["team_goals"].shift(1).rolling(5).mean()
        team_df["rolling_conceded_5"] = team_df["opp_goals"].shift(1).rolling(5).mean()
        
        # Days since last match
        team_df["days_since_last"] = team_df["date"].diff().dt.days
        
        # Keep only necessary columns to merge back
        team_df = team_df[["date", "home_team", "away_team", "rolling_ppg_5", "rolling_goals_5", "rolling_conceded_5", "days_since_last"]]
        team_df["team"] = team
        team_stats.append(team_stats_dict := team_df.to_dict("records"))

    # Convert back to flat list and then DataFrame for faster lookup
    flat_stats = [item for sublist in team_stats for item in sublist]
    stats_df = pd.DataFrame(flat_stats)
    
    # Merge back to original df
    # For Home Team
    df = df.merge(
        stats_df, 
        left_on=["date", "home_team", "away_team", "home_team"], 
        right_on=["date", "home_team", "away_team", "team"],
        how="left",
        suffixes=("", "_home")
    ).drop(columns=["team"])
    
    # For Away Team
    df = df.merge(
        stats_df, 
        left_on=["date", "home_team", "away_team", "away_team"], 
        right_on=["date", "home_team", "away_team", "team"],
        how="left",
        suffixes=("_home", "_away")
    ).drop(columns=["team"])
    
    # H2H Features (Last 5 meetings)
    # This is more expensive, but worth it for a "Tier 1" project
    # We'll skip it for now if it's too slow, or use a faster implementation
    
    # Fill NaNs from rolling (first few games for each team)
    df = df.fillna(0)
    
    return df

def objective(trial, X, y):
    param = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
    }
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    losses = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train, y_train)
        
        preds = model.predict_proba(X_val)
        losses.append(log_loss(y_val, preds))
        
    return np.mean(losses)

def main():
    print("=" * 60)
    print("  WORLD CUP LAB - Model Training Pipeline")
    print("=" * 60)
    
    df = pd.read_parquet(PROCESSED / "results_with_elo.parquet")
    
    # Feature Engineering
    df = engineer_features(df)
    
    # Features to use
    features = [
        "elo_diff", "home_elo_before", "away_elo_before", 
        "rolling_ppg_5_home", "rolling_goals_5_home", "rolling_conceded_5_home", "days_since_last_home",
        "rolling_ppg_5_away", "rolling_goals_5_away", "rolling_conceded_5_away", "days_since_last_away",
        "neutral", "tournament_weight", "is_knockout", "confederation_clash"
    ]
    
    X = df[features]
    y = df["result_code"] # 0: away_win, 1: draw, 2: home_win
    
    # Categorical features
    X["neutral"] = X["neutral"].astype(int)
    X["is_knockout"] = X["is_knockout"].astype(int)
    
    # Train/Test Split (Recent matches for testing)
    # Let's say matches after 2022-01-01 are for final evaluation
    train_mask = df["date"] < "2022-01-01"
    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    
    print(f"  Training on {len(X_train_full):,} matches")
    print(f"  Testing on {len(X_test):,} matches")
    
    # Hyperparameter Tuning with Optuna
    print("\n  Optimizing hyperparameters with Optuna ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_train_full, y_train_full), n_trials=20)
    
    print(f"  Best trial: {study.best_trial.params}")
    
    # Train Final Model with Calibration
    print("\n  Training final calibrated model ...")
    base_model = lgb.LGBMClassifier(**study.best_params, verbosity=-1)
    
    calibrated_model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    calibrated_model.fit(X_train_full, y_train_full)
    
    # Evaluation
    preds_proba = calibrated_model.predict_proba(X_test)
    preds = calibrated_model.predict(X_test)
    
    ll = log_loss(y_test, preds_proba)
    # ROC-AUC multi-class
    auc = roc_auc_score(y_test, preds_proba, multi_class="ovr", average="macro")
    
    print(f"\n  -- Evaluation Results (Test Set) --")
    print(f"  Log Loss: {ll:.4f}")
    print(f"  ROC-AUC:  {auc:.4f}")
    
    # Save Model
    joblib.dump(calibrated_model, MODELS / "lgbm_match_predictor.joblib")
    joblib.dump(features, MODELS / "feature_names.joblib")
    
    print(f"\n  [OK] Model saved to models/lgbm_match_predictor.joblib")
    
    print("\n" + "=" * 60)
    print("  Model training complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
