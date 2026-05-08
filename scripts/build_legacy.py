"""
Phase 5: Player Legacy Index
Computes era-adjusted legacy scores for every international goalscorer.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Paths
BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "dataset" / "processed"
OUT = PROCESSED

def compute_legacy_score(row):
    """
    Legacy_Score = base_points * opponent_elo_multiplier * tournament_weight * clutch_multiplier
    """
    # Base points
    if row["own_goal"]:
        return -0.5 # Deduct for own goals
    
    base = 1.0
    if row["penalty"]:
        base = 0.7
        
    # Opponent ELO multiplier
    # Score vs a 2000 ELO team is worth more than vs a 1500 ELO team
    # Multiplier = (ELO_opponent / 1500)^0.5
    opp_elo = row["opponent_elo_before"]
    elo_mult = (opp_elo / 1500.0) ** 0.5
    
    # Tournament weight (1-5)
    t_weight = row["tournament_weight"]
    
    # Clutch multiplier
    # Minute based
    minute_mult = 1.0
    if row["minute"] >= 80:
        minute_mult = 1.2
    
    # Score state multiplier (requires match context)
    # For now we'll stick to minute as a proxy for clutch if full context is missing
    # or if we have goal order we can do better.
    
    return base * elo_mult * t_weight * minute_mult

def main():
    print("=" * 60)
    print("  WORLD CUP LAB - Player Legacy Pipeline")
    print("=" * 60)
    
    # Load data
    print("  Loading goalscorers and match results ...")
    goals = pd.read_parquet(PROCESSED / "goalscorers_cleaned.parquet")
    results = pd.read_parquet(PROCESSED / "results_with_elo.parquet")
    
    # Merge to get opponent ELO and tournament weight for each goal
    # We need to know which team the goalscorer belongs to, to identify the opponent
    
    # Join goals with results
    # Each goal has: date, home_team, away_team, team (of scorer)
    print("  Joining goals with match context ...")
    df = goals.merge(
        results[["date", "home_team", "away_team", "home_elo_before", "away_elo_before", "tournament_weight"]],
        on=["date", "home_team", "away_team"],
        how="left"
    )
    
    # Identify opponent ELO
    df["opponent_elo_before"] = np.where(
        df["team"] == df["home_team"], 
        df["away_elo_before"], 
        df["home_elo_before"]
    )
    
    # Fill missing ELO with default 1500
    df["opponent_elo_before"] = df["opponent_elo_before"].fillna(1500.0)
    df["tournament_weight"] = df["tournament_weight"].fillna(1)
    
    # Compute Legacy Score
    print("  Computing legacy scores for each goal ...")
    df["legacy_score"] = df.apply(compute_legacy_score, axis=1)
    
    # Aggregate per player
    print("  Aggregating player stats ...")
    player_legacy = df.groupby(["scorer", "team"]).agg(
        total_goals=("scorer", "count"),
        total_legacy_score=("legacy_score", "sum"),
        avg_opponent_elo=("opponent_elo_before", "mean"),
        penalties=("penalty", "sum"),
        own_goals=("own_goal", "sum")
    ).reset_index()
    
    # Filter out own goals for the main ranking (already handled in formula but for clarity)
    # Sort by legacy score
    player_legacy = player_legacy.sort_values("total_legacy_score", ascending=False).reset_index(drop=True)
    
    # Save results
    player_legacy.to_parquet(OUT / "player_legacy_index.parquet", index=False)
    print(f"  [OK] Saved player_legacy_index.parquet ({len(player_legacy):,} players)")
    
    # Print top 10
    print("\n  -- All-Time Player Legacy Top 10 --")
    print(player_legacy[["scorer", "team", "total_goals", "total_legacy_score"]].head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("  Legacy computation complete.")
    print("=" * 60)

if __name__ == "__main__":
    main()
