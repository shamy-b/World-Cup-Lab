import joblib
import numpy as np
print("Loading saved models...")
xgb_model = joblib.load("models/xgb_tuned.joblib")
lgb_model = joblib.load("models/lgb_tuned.joblib")
feature_cols = joblib.load("models/feature_cols.joblib")
print(f"Features used: {len(feature_cols)}")
print(f"XGBoost params: n_estimators={xgb_model.n_estimators}, max_depth={xgb_model.max_depth}, lr={xgb_model.learning_rate}")
print(f"LightGBM params: n_estimators={lgb_model.n_estimators}, max_depth={lgb_model.max_depth}, lr={lgb_model.learning_rate}")
print("Models loaded successfully!")
