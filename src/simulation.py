import pandas as pd
import numpy as np
import joblib
import sys
import os
from tqdm import tqdm
from itertools import combinations

sys.path.insert(0, 'src')
from ingestion import ChronosIngestor
from features import FeatureEngineer
from elo import EloSystem

class WorldCupSimulator:
    def __init__(self):
        print("[*] Initializing 2026 World Cup Monte Carlo Simulator...")
        self.xgb_model = joblib.load("models/xgb_tuned.joblib")
        self.lgb_model = joblib.load("models/lgb_tuned.joblib")
        self.meta_model = joblib.load("models/meta_ensemble.joblib")
        self.feature_cols = joblib.load("models/feature_cols.joblib")
        
        self.team_features = {}
        self.prob_matrix = {} # {(team1, team2): p_team1_advances}
        self._prepare_latest_features()

    def _prepare_latest_features(self):
        """Extracts the most recent feature state for every team."""
        print("[*] Extracting current team states...")
        ingestor = ChronosIngestor()
        ingestor.load_all()
        df = ingestor.results
        
        fe = FeatureEngineer()
        df = fe.generate_all_features(df)
        
        self.elo_system = EloSystem()
        df = self.elo_system.process_history(df)
        
        # We need the last known rolling features for each team
        # Since rolling features are attached to both home and away, we can just find the last match for each team
        latest_records = {}
        
        # We need to reconstruct team-specific features. 
        # The easiest way is to look at the 'flat' dataframe logic or just extract from df
        for team in pd.concat([df['home_team'], df['away_team']]).unique():
            # Get last match where team was home
            home_matches = df[df['home_team'] == team].tail(1)
            away_matches = df[df['away_team'] == team].tail(1)
            
            last_date_home = home_matches['date'].values[0] if not home_matches.empty else np.datetime64('1900-01-01')
            last_date_away = away_matches['date'].values[0] if not away_matches.empty else np.datetime64('1900-01-01')
            
            features = {}
            if last_date_home > last_date_away:
                row = home_matches.iloc[0]
                features['elo'] = row['elo_home']
                for c in self.feature_cols:
                    if c.endswith('_home') and not c.startswith('elo_'): 
                        features[c.replace('_home', '')] = row[c]
            elif not away_matches.empty:
                row = away_matches.iloc[0]
                features['elo'] = row['elo_away']
                for c in self.feature_cols:
                    if c.endswith('_away') and not c.startswith('elo_'): 
                        features[c.replace('_away', '')] = row[c]
            
            if 'elo' in features:
                self.team_features[team] = features

    def _predict_matchup(self, team1, team2):
        """Predicts probability of team1 advancing against team2 at a neutral venue."""
        if (team1, team2) in self.prob_matrix:
            return self.prob_matrix[(team1, team2)]
            
        f1 = self.team_features.get(team1)
        f2 = self.team_features.get(team2)
        
        if not f1 or not f2:
            return 0.5 # Fallback
            
        # Construct feature row (Team 1 as Home, Team 2 as Away)
        row1 = {
            'elo_home': f1['elo'], 'elo_away': f2['elo'], 'elo_diff': f1['elo'] - f2['elo'],
            'elo_expected_home': 1 / (1 + 10 ** ((f2['elo'] - f1['elo']) / 400)),
            'tournament_prestige': 1.0, # World Cup
            'is_neutral': 1,
            # H2H (fallback to 0.5/0.0 if not easily available)
            'h2h_home_win_rate': 0.5, 'h2h_away_win_rate': 0.5, 'h2h_draw_rate': 0.0, 'h2h_matches': 0
        }
        row1['elo_expected_away'] = 1 - row1['elo_expected_home']
        
        for c in self.feature_cols:
            if c.endswith('_home') and not c.startswith('h2h') and not c.startswith('elo'):
                row1[c] = f1.get(c.replace('_home', ''), 0)
            elif c.endswith('_away') and not c.startswith('h2h') and not c.startswith('elo'):
                row1[c] = f2.get(c.replace('_away', ''), 0)
                
        # Fill missing with 0
        for c in self.feature_cols:
            if c not in row1: row1[c] = 0
            
        X = pd.DataFrame([row1])[self.feature_cols]
        
        xgb_p = self.xgb_model.predict_proba(X)
        lgb_p = self.lgb_model.predict_proba(X)
        meta_X = np.hstack([xgb_p, lgb_p])
        probs = self.meta_model.predict_proba(meta_X)[0] # [Away, Draw, Home]
        
        # Probability Team 1 (Home) wins + 50% chance of winning a draw (penalties)
        p_t1_advances = probs[2] + (probs[1] * 0.5)
        
        self.prob_matrix[(team1, team2)] = p_t1_advances
        self.prob_matrix[(team2, team1)] = 1 - p_t1_advances
        
        return p_t1_advances

    def precalculate_matrix(self, teams):
        """Precalculates all possible matchups for lightning-fast Monte Carlo."""
        print(f"[*] Precalculating probability matrix for {len(teams)} teams...")
        for t1, t2 in combinations(teams, 2):
            self._predict_matchup(t1, t2)
            
    def run_monte_carlo(self, teams, iterations=10000):
        self.precalculate_matrix(teams)
        
        print(f"\n[*] Running {iterations} Monte Carlo simulations...")
        
        results = {t: {'R16': 0, 'QF': 0, 'SF': 0, 'Final': 0, 'Winner': 0} for t in teams}
        
        for _ in tqdm(range(iterations), desc="Simulating Brackets"):
            # Round of 32
            r16 = []
            for i in range(0, 32, 2):
                t1, t2 = teams[i], teams[i+1]
                p = self.prob_matrix[(t1, t2)]
                winner = t1 if np.random.random() < p else t2
                r16.append(winner)
                results[winner]['R16'] += 1
                
            # Quarter Finals
            qf = []
            for i in range(0, 16, 2):
                t1, t2 = r16[i], r16[i+1]
                p = self.prob_matrix[(t1, t2)]
                winner = t1 if np.random.random() < p else t2
                qf.append(winner)
                results[winner]['QF'] += 1
                
            # Semi Finals
            sf = []
            for i in range(0, 8, 2):
                t1, t2 = qf[i], qf[i+1]
                p = self.prob_matrix[(t1, t2)]
                winner = t1 if np.random.random() < p else t2
                sf.append(winner)
                results[winner]['SF'] += 1
                
            # Final
            final = []
            for i in range(0, 4, 2):
                t1, t2 = sf[i], sf[i+1]
                p = self.prob_matrix[(t1, t2)]
                winner = t1 if np.random.random() < p else t2
                final.append(winner)
                results[winner]['Final'] += 1
                
            # Winner
            t1, t2 = final[0], final[1]
            p = self.prob_matrix[(t1, t2)]
            champ = t1 if np.random.random() < p else t2
            results[champ]['Winner'] += 1
            
        # Format report
        report = []
        for t in teams:
            report.append({
                'Team': t,
                'R16 (%)': (results[t]['R16'] / iterations) * 100,
                'QF (%)': (results[t]['QF'] / iterations) * 100,
                'SF (%)': (results[t]['SF'] / iterations) * 100,
                'Final (%)': (results[t]['Final'] / iterations) * 100,
                'Win (%)': (results[t]['Winner'] / iterations) * 100,
            })
            
        df_report = pd.DataFrame(report).sort_values('Win (%)', ascending=False).reset_index(drop=True)
        return df_report

if __name__ == "__main__":
    # Top 32 teams for a mock World Cup knockout stage
    top_32 = [
        "Spain", "Argentina", "France", "England", "Netherlands", "Colombia", "Germany", "Brazil",
        "Portugal", "Japan", "Uruguay", "Croatia", "Italy", "Morocco", "Switzerland", "Senegal",
        "United States", "Mexico", "Iran", "South Korea", "Denmark", "Austria", "Ecuador", "Ukraine",
        "Australia", "Peru", "Serbia", "Poland", "Sweden", "Wales", "Hungary", "Ivory Coast"
    ]
    
    # Shuffle for a random bracket draw
    np.random.seed(42)
    np.random.shuffle(top_32)
    
    sim = WorldCupSimulator()
    report = sim.run_monte_carlo(top_32, iterations=10000)
    
    print("\n" + "="*80)
    print("2026 WORLD CUP MONTE CARLO PROJECTIONS (10,000 SIMULATIONS)")
    print("="*80)
    print(report.to_string(index=False, float_format="%.1f"))
    
    # Save to artifact directory if needed
    report.to_csv("simulation_report.csv", index=False)
