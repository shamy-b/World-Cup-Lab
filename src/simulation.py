import pandas as pd
import numpy as np
import joblib
from elo import EloSystem
from features import FeatureEngineer

class TournamentSimulator:
    """
    Monte Carlo Simulation Engine for Chronos.
    Simulates match outcomes and tournament progression using probabilistic modeling.
    """
    def __init__(self, model_path="models/baseline_xgb.joblib"):
        self.model = joblib.load(model_path)
        self.elo_system = EloSystem()
        self.fe = FeatureEngineer()
        
        # Load latest data to get current ratings
        from ingestion import ChronosIngestor
        ingestor = ChronosIngestor()
        ingestor.load_all()
        self.elo_system.process_history(ingestor.results)

    def simulate_match(self, home_team, away_team, tournament="FIFA World Cup", neutral=True):
        """
        Simulates a single match outcome between two teams.
        Returns: Probabilities for [Away, Draw, Home]
        """
        # Prepare features for this match
        elo_h = self.elo_system.get_rating(home_team)
        elo_a = self.elo_system.get_rating(away_team)
        
        # We'll use current Elo and average rolling stats for simulation
        # In a real system, we'd pull the actual last rolling stats for these teams
        features = pd.DataFrame([{
            'elo_home': elo_h,
            'elo_away': elo_a,
            'elo_diff': elo_h - elo_a,
            'rolling_win_rate_5_home': 0.5, # Defaulting for simulation simplicity
            'rolling_win_rate_5_away': 0.5,
            'rolling_win_rate_10_home': 0.5,
            'rolling_win_rate_10_away': 0.5,
            'rolling_gf_5_home': 1.5,
            'rolling_ga_5_home': 1.0,
            'rolling_gf_5_away': 1.5,
            'rolling_ga_5_away': 1.0
        }])
        
        probs = self.model.predict_proba(features)[0]
        return probs

    def simulate_knockout(self, team1, team2):
        """
        Simulates a knockout match where a winner is mandatory.
        Uses match probabilities + tie-breaking logic.
        """
        probs = self.simulate_match(team1, team2)
        # Probabilities: [Away, Draw, Home]
        p_away, p_draw, p_home = probs
        
        # If draw, simulate a coin flip (penalty shootout)
        # In the blueprint, we'd use a specific shootout model
        outcome = np.random.choice(['away', 'draw', 'home'], p=[p_away, p_draw, p_home])
        
        if outcome == 'home':
            return team1
        elif outcome == 'away':
            return team2
        else:
            # Penalties: 50/50 for now
            return team1 if np.random.random() > 0.5 else team2

    def simulate_tournament(self, teams):
        """
        Simulates a simple knockout tournament (must be power of 2 teams).
        """
        current_round = teams
        round_num = 1
        
        while len(current_round) > 1:
            print(f"\n--- Round {round_num} ---")
            next_round = []
            for i in range(0, len(current_round), 2):
                t1, t2 = current_round[i], current_round[i+1]
                winner = self.simulate_knockout(t1, t2)
                print(f"  {t1} vs {t2} -> Winner: {winner}")
                next_round.append(winner)
            current_round = next_round
            round_num += 1
            
        return current_round[0]

if __name__ == "__main__":
    sim = TournamentSimulator()
    
    # Simulate a "Mini World Cup"
    wc_teams = ["Argentina", "France", "Brazil", "England", "Spain", "Netherlands", "Portugal", "Germany"]
    winner = sim.simulate_tournament(wc_teams)
    print(f"\n[🏆] TOURNAMENT WINNER: {winner}")
