import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, log_loss, classification_report, f1_score
from ingestion import ChronosIngestor
from features import FeatureEngineer
from elo import EloSystem

# Load models
xgb_model = joblib.load("models/xgb_tuned.joblib")
lgb_model = joblib.load("models/lgb_tuned.joblib")
meta_model = joblib.load("models/meta_ensemble.joblib")
feature_cols = joblib.load("models/feature_cols.joblib")

# Rebuild data
ingestor = ChronosIngestor()
ingestor.load_all()
df = ingestor.results

fe = FeatureEngineer()
df = fe.generate_all_features(df)

elo = EloSystem()
df = elo.process_history(df)

df['elo_expected_home'] = 1 / (1 + 10 ** ((df['elo_away'] - df['elo_home']) / 400))
df['elo_expected_away'] = 1 - df['elo_expected_home']
df['target'] = df['outcome'] + 1
df = df.dropna(subset=feature_cols)

test = df[df['date'] >= '2022-01-01']
X_test = test[feature_cols]
y_test = test['target']

# Predictions
xgb_probs = xgb_model.predict_proba(X_test)
lgb_probs = lgb_model.predict_proba(X_test)
xgb_preds = xgb_model.predict(X_test)
lgb_preds = lgb_model.predict(X_test)
meta_X = np.hstack([xgb_probs, lgb_probs])
ens_probs = meta_model.predict_proba(meta_X)
ens_preds = meta_model.predict(meta_X)
avg_probs = (xgb_probs + lgb_probs) / 2
avg_preds = np.argmax(avg_probs, axis=1)

print("=" * 60)
print("CHRONOS ENGINE v2 - EVALUATION REPORT")
print("=" * 60)
print(f"Test set: {len(test)} matches (2022+)")
print(f"Features: {len(feature_cols)}")
print()
print(f"{'Model':<25} {'Accuracy':>10} {'LogLoss':>10} {'F1_Macro':>10}")
print("-" * 55)
for name, preds, probs in [
    ("XGBoost (Tuned)", xgb_preds, xgb_probs),
    ("LightGBM (Tuned)", lgb_preds, lgb_probs),
    ("Average Ensemble", avg_preds, avg_probs),
    ("Stacking Ensemble", ens_preds, ens_probs),
]:
    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs)
    f1 = f1_score(y_test, preds, average='macro')
    print(f"{name:<25} {acc:>10.4f} {ll:>10.4f} {f1:>10.4f}")

print()
print("=" * 60)
print("DETAILED REPORT - Stacking Ensemble")
print("=" * 60)
print(classification_report(y_test, ens_preds, target_names=['Away', 'Draw', 'Home']))

print("=" * 60)
print("v1 vs v2 COMPARISON")
print("=" * 60)
best_acc = accuracy_score(y_test, ens_preds)
best_ll = log_loss(y_test, ens_probs)
print(f"  OLD: Accuracy=0.5868  LogLoss=0.8988  DrawRecall=0.04")
print(f"  NEW: Accuracy={best_acc:.4f}  LogLoss={best_ll:.4f}")
print(f"  Improvement: {(best_acc - 0.5868)*100:+.1f}pp accuracy")

print()
print("Top 10 Teams by Elo:")
top = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)[:10]
for i, (team, rating) in enumerate(top, 1):
    print(f"  {i:>2}. {team:<20} {rating:.1f}")
