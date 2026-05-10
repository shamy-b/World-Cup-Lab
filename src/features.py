import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Advanced Feature Engineering Engine for Chronos.
    Generates rolling statistics, head-to-head records, momentum,
    draw tendencies, tournament prestige, and historical context features.
    """
    
    # Tournament prestige scores (normalized 0-1)
    TOURNAMENT_PRESTIGE = {
        'FIFA World Cup': 1.0,
        'Confederations Cup': 0.85,
        'UEFA Euro': 0.9,
        'Copa América': 0.85,
        'African Cup of Nations': 0.8,
        'AFC Asian Cup': 0.8,
        'Gold Cup': 0.75,
        'FIFA World Cup qualification': 0.7,
        'UEFA Euro qualification': 0.65,
        'UEFA Nations League': 0.6,
        'CONCACAF Nations League': 0.55,
        'African Cup of Nations qualification': 0.55,
        'AFC Asian Cup qualification': 0.55,
        'Olympic Games': 0.5,
        'Friendly': 0.2,
    }
    
    def __init__(self):
        self._h2h_cache = {}

    def _get_prestige(self, tournament):
        """Returns prestige score with keyword fallback."""
        if tournament in self.TOURNAMENT_PRESTIGE:
            return self.TOURNAMENT_PRESTIGE[tournament]
        t_lower = tournament.lower()
        if 'world cup' in t_lower:
            return 0.7
        elif 'qualification' in t_lower:
            return 0.5
        elif 'cup' in t_lower or 'championship' in t_lower:
            return 0.45
        elif 'games' in t_lower:
            return 0.4
        elif 'friendly' in t_lower:
            return 0.2
        return 0.35

    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Master function: generates all features in one pass.
        """
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # === BASIC FEATURES ===
        df['outcome'] = np.where(df['home_score'] > df['away_score'], 1,
                        np.where(df['home_score'] < df['away_score'], -1, 0))
        df['goal_diff'] = df['home_score'] - df['away_score']
        df['total_goals'] = df['home_score'] + df['away_score']
        
        # === TOURNAMENT PRESTIGE ===
        df['tournament_prestige'] = df['tournament'].apply(self._get_prestige)
        
        # === NEUTRAL VENUE ===
        df['is_neutral'] = df['neutral'].astype(int)
        
        # === FLATTEN TO TEAM-LEVEL FOR ROLLING STATS ===
        print("[*] Generating advanced rolling features...")
        team_matches = []
        for _, row in df.iterrows():
            base = {
                'date': row['date'],
                'match_id': row['match_id'],
                'goal_diff_match': row['goal_diff'],
                'total_goals_match': row['total_goals'],
            }
            # Home perspective
            h = base.copy()
            h.update({
                'team': row['home_team'],
                'opponent': row['away_team'],
                'goals_for': row['home_score'],
                'goals_against': row['away_score'],
                'is_home': 1,
                'outcome': row['outcome'],
                'is_draw': 1 if row['outcome'] == 0 else 0,
                'clean_sheet': 1 if row['away_score'] == 0 else 0,
            })
            team_matches.append(h)
            # Away perspective
            a = base.copy()
            a.update({
                'team': row['away_team'],
                'opponent': row['home_team'],
                'goals_for': row['away_score'],
                'goals_against': row['home_score'],
                'is_home': 0,
                'outcome': -row['outcome'],
                'is_draw': 1 if row['outcome'] == 0 else 0,
                'clean_sheet': 1 if row['home_score'] == 0 else 0,
            })
            team_matches.append(a)
            
        flat = pd.DataFrame(team_matches).sort_values(['team', 'date'])
        
        # === ROLLING WINDOWS ===
        windows = [3, 5, 10, 20]
        for w in windows:
            g = flat.groupby('team')
            
            # Goals
            flat[f'roll_gf_{w}'] = g['goals_for'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            flat[f'roll_ga_{w}'] = g['goals_against'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            
            # Win rate
            flat[f'roll_win_{w}'] = g['outcome'].transform(
                lambda x: (x.shift(1) == 1).astype(float).rolling(w, min_periods=1).mean())
            
            # Draw rate
            flat[f'roll_draw_{w}'] = g['is_draw'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            
            # Clean sheet rate
            flat[f'roll_cs_{w}'] = g['clean_sheet'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            
            # Goal difference variance (unpredictability)
            flat[f'roll_gd_std_{w}'] = g['goals_for'].transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).std())
        
        # === MOMENTUM (weighted recent form: W=3pts, D=1pt, L=0pt) ===
        flat['points'] = flat['outcome'].map({1: 3, 0: 1, -1: 0})
        flat['momentum_5'] = flat.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        flat['momentum_10'] = flat.groupby('team')['points'].transform(
            lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        
        # === DAYS SINCE LAST MATCH (rest/fatigue) ===
        flat['days_rest'] = flat.groupby('team')['date'].diff().dt.days.fillna(30)
        flat['days_rest'] = flat['days_rest'].clip(upper=365)

        # === MERGE BACK ===
        # Define which rolling columns to keep
        roll_cols = [c for c in flat.columns if c.startswith('roll_') or c in ['momentum_5', 'momentum_10', 'days_rest']]
        
        home_stats = flat[flat['is_home'] == 1][['date', 'team'] + roll_cols].copy()
        away_stats = flat[flat['is_home'] == 0][['date', 'team'] + roll_cols].copy()
        
        df = df.merge(home_stats, left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').drop(columns='team')
        df = df.merge(away_stats, left_on=['date', 'away_team'], right_on=['date', 'team'], how='left', suffixes=('_home', '_away')).drop(columns='team')
        
        # === HEAD-TO-HEAD FEATURES ===
        print("[*] Generating head-to-head features...")
        df = self._generate_h2h_features(df)
        
        print(f"[+] Feature engineering complete. Total features: {len(df.columns)}")
        return df

    def _generate_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates head-to-head historical record between each pair.
        Uses only matches BEFORE the current one (no leakage).
        """
        h2h_home_wins = []
        h2h_away_wins = []
        h2h_draws = []
        h2h_total = []
        
        # Build a lookup: for each pair, track cumulative results
        pair_stats = {}
        
        for _, row in df.iterrows():
            home = row['home_team']
            away = row['away_team']
            key = tuple(sorted([home, away]))
            
            if key not in pair_stats:
                pair_stats[key] = {'wins': {home: 0, away: 0}, 'draws': 0, 'total': 0}
            
            stats = pair_stats[key]
            total = stats['total']
            
            if total == 0:
                h2h_home_wins.append(0.5)
                h2h_away_wins.append(0.5)
                h2h_draws.append(0.0)
                h2h_total.append(0)
            else:
                h_wins = stats['wins'].get(home, 0)
                a_wins = stats['wins'].get(away, 0)
                h2h_home_wins.append(h_wins / total)
                h2h_away_wins.append(a_wins / total)
                h2h_draws.append(stats['draws'] / total)
                h2h_total.append(total)
            
            # Update AFTER recording
            outcome = row['outcome']
            stats['total'] += 1
            if outcome == 1:
                stats['wins'][home] = stats['wins'].get(home, 0) + 1
            elif outcome == -1:
                stats['wins'][away] = stats['wins'].get(away, 0) + 1
            else:
                stats['draws'] += 1
        
        df['h2h_home_win_rate'] = h2h_home_wins
        df['h2h_away_win_rate'] = h2h_away_wins
        df['h2h_draw_rate'] = h2h_draws
        df['h2h_matches'] = h2h_total
        
        return df

if __name__ == "__main__":
    from ingestion import ChronosIngestor
    
    ingestor = ChronosIngestor()
    ingestor.load_all()
    
    fe = FeatureEngineer()
    df = fe.generate_all_features(ingestor.results)
    
    print(f"\n[+] Columns ({len(df.columns)}):")
    print(df.columns.tolist())
    print(df[['date', 'home_team', 'away_team', 'h2h_home_win_rate', 'momentum_5_home']].tail())
