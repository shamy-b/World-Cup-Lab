# World Cup Lab — Implementation Plan
## The Definitive 150-Year International Football Intelligence Platform

### Stack: Python | Streamlit | LightGBM | SHAP | Optuna | Plotly | NumPy | SciPy

---

## Phase 0 — Environment Setup (1 hr)
- [ ] Pin all dependencies in requirements.txt (streamlit, lgbm, shap, optuna, plotly, pyarrow)
- [ ] Verify CSV integrity and date parsing
- [ ] Create dataset/processed/ pipeline

## Phase 1 — Data Engineering (2 hrs)
- [ ] Normalize all team names via former_names.csv (Zaire→DR Congo, Soviet Union→Russia, etc.)
- [ ] Engineer: goal_diff, result, decade, is_knockout, confederation columns
- [ ] Link goalscorers.csv to match results (match_goal_number)
- [ ] Save results_cleaned.parquet and goalscorers_cleaned.parquet

## Phase 2 — ELO Engine (3 hrs)
- [ ] ELOEngine class with dynamic K-factor (WC=60, Continental=50, Qualifier=40, Friendly=20)
- [ ] Goal difference multiplier: ln(|gd|+1)*0.4+1.0 (capped at 1.75)
- [ ] Process all 49K matches chronologically
- [ ] Save full ELO timeline to results_with_elo.parquet

## Phase 3 — ML Match Predictor (4 hrs)
- [ ] 16-feature matrix: elo_diff, h2h_winrate, form_5, rolling goals, rest_days, etc.
- [ ] LightGBM multi-class (home_win/draw/away_win) with Optuna (50 trials)
- [ ] CalibratedClassifierCV + SHAP TreeExplainer
- [ ] Save models/lgbm_match_predictor.joblib

## Phase 4 — Monte Carlo Simulator (3 hrs)
- [ ] 50,000 bracket simulations using calibrated ML model
- [ ] Bayesian penalty shootout model from shootouts.csv
- [ ] Output: championship probabilities + Sankey round-survival diagram

## Phase 5 — Player Legacy Index (2 hrs)
- [ ] Legacy Score = base_pts * opponent_elo_multiplier * tournament_weight * clutch_multiplier
- [ ] Era-adjusted ranking of all ~6,000 unique scorers
- [ ] Animated bar chart race by decade

## Phase 6 — Streamlit App UI (4 hrs)
- [ ] Dark glassmorphism theme (bg #0a0a0f, accent #00d4ff)
- [ ] Home: animated counters + ELO choropleth globe
- [ ] ELO Rankings: live table + sparklines + H2H battle mode
- [ ] Match Predictor: probability gauges + SHAP waterfall
- [ ] Tournament Simulator: live progress + Sankey diagram
- [ ] Player Legacy: leaderboard + career deep dive + bar chart race

## Definition of Done
- [ ] All 5 pages functional with real data
- [ ] ELO engine covers all 49,289 matches
- [ ] ML model: Log Loss < 0.95 | ROC-AUC > 0.72 | Brier < 0.22
- [ ] 50K simulations run in < 12 seconds
- [ ] Deployable to Streamlit Community Cloud
