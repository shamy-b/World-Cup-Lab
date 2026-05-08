"""
Phase 4: Monte Carlo Tournament Simulator
Core logic for simulating brackets based on the ML predictor.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Paths
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "dataset" / "processed"
MODELS = BASE / "models"

class TournamentSimulator:
    def __init__(self, model_path, feature_names_path, current_elo_path, shootout_stats_path):
        self.model = joblib.load(model_path)
        self.features = joblib.load(feature_names_path)
        self.current_elo = pd.read_parquet(current_elo_path).set_index("team")["elo"].to_dict()
        self.shootout_stats = pd.read_parquet(shootout_stats_path).set_index("team")["smoothed_win_rate"].to_dict()
        self.rng = np.random.default_rng()

    def get_elo(self, team):
        return self.current_elo.get(team, 1500.0)

    def predict_match(self, team_h, team_a, neutral=True, weight=5, is_knockout=True):
        elo_h = self.get_elo(team_h)
        elo_a = self.get_elo(team_a)
        
        # Build feature vector (simplified for simulation - using current ELO as primary driver)
        # In a real sim, we'd ideally have rolling form, but current ELO is a strong proxy.
        # For simplicity in this script, we'll set rolling features to 0 or averages.
        
        feat_dict = {
            "elo_diff": elo_h - elo_a,
            "home_elo_before": elo_h,
            "away_elo_before": elo_a,
            "rolling_ppg_5_home": 1.5, # Assume decent form
            "rolling_goals_5_home": 1.5,
            "rolling_conceded_5_home": 1.0,
            "days_since_last_home": 4,
            "rolling_ppg_5_away": 1.5,
            "rolling_goals_5_away": 1.5,
            "rolling_conceded_5_away": 1.0,
            "days_since_last_away": 4,
            "neutral": int(neutral),
            "tournament_weight": weight,
            "is_knockout": int(is_knockout),
            "confederation_clash": 1 # Assume different confederations for WC
        }
        
        X = pd.DataFrame([feat_dict])[self.features]
        probs = self.model.predict_proba(X)[0] # [P(away_win), P(draw), P(home_win)]
        
        return probs

    def simulate_match(self, team_h, team_a, is_knockout=True):
        probs = self.predict_match(team_h, team_a, is_knockout=is_knockout)
        
        # Stochastic outcome
        outcome = self.rng.choice([0, 1, 2], p=probs)
        
        if outcome == 2: return team_h
        if outcome == 0: return team_a
        
        # If draw and knockout -> Shootouts
        if is_knockout:
            win_rate_h = self.shootout_stats.get(team_h, 0.5)
            win_rate_a = self.shootout_stats.get(team_a, 0.5)
            
            # Relative win prob in shootout
            prob_h = win_rate_h / (win_rate_h + win_rate_a)
            return team_h if self.rng.random() < prob_h else team_a
        
        return "Draw"

    def simulate_tournament(self, teams, n_sims=1000):
        """Simulate a simple knockout tournament (must be power of 2 teams)."""
        winners = []
        for _ in range(n_sims):
            current_round = list(teams)
            while len(current_round) > 1:
                next_round = []
                for i in range(0, len(current_round), 2):
                    winner = self.simulate_match(current_round[i], current_round[i+1])
                    next_round.append(winner)
                current_round = next_round
            winners.append(current_round[0])
            
        return pd.Series(winners).value_counts(normalize=True)

def main():
    print("=" * 60)
    print("  WORLD CUP LAB - Tournament Simulator Test")
    print("=" * 60)
    
    # Check if model exists
    if not (MODELS / "lgbm_match_predictor.joblib").exists():
        print("  [ERROR] Model not found. Run train_model.py first.")
        return

    sim = TournamentSimulator(
        MODELS / "lgbm_match_predictor.joblib",
        MODELS / "feature_names.joblib",
        PROCESSED / "current_elo_ratings.parquet",
        PROCESSED / "shootout_stats.parquet"
    )
    
    # Mock Round of 16
    wc_teams = [
        "Argentina", "France", "Brazil", "England", 
        "Spain", "Netherlands", "Portugal", "Germany",
        "Belgium", "Croatia", "Uruguay", "Italy",
        "Morocco", "Japan", "United States", "Mexico"
    ]
    
    print(f"  Simulating a mock Round of 16 bracket with {len(wc_teams)} teams ...")
    results = sim.simulate_tournament(wc_teams, n_sims=1000)
    
    print("\n  -- Championship Probabilities (1,000 sims) --")
    print(results.head(10).to_string())
    
    print("\n" + "=" * 60)
    print("  Simulation complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
