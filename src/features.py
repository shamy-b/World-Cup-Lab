import pandas as pd
import numpy as np

class FeatureEngineer:
    """
    Feature Engineering Engine for Chronos.
    Generates rolling statistics, momentum, and historical context features.
    """
    
    def __init__(self):
        pass

    def generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates simple match-level features.
        """
        df = df.copy()
        
        # Match Outcome (Target)
        # 1: Home Win, 0: Draw, -1: Away Win
        df['outcome'] = np.where(df['home_score'] > df['away_score'], 1,
                        np.where(df['home_score'] < df['away_score'], -1, 0))
        
        # Goal Difference
        df['goal_diff'] = df['home_score'] - df['away_score']
        
        # Total Goals
        df['total_goals'] = df['home_score'] + df['away_score']
        
        return df

    def generate_rolling_stats(self, df: pd.DataFrame, windows=[5, 10, 20]) -> pd.DataFrame:
        """
        Calculates rolling averages for teams.
        This is complex because a team can be either 'home' or 'away'.
        """
        df = df.sort_values('date')
        
        # To calculate rolling stats, we need to flatten the matches so each team-match is a row
        team_matches = []
        for _, row in df.iterrows():
            # Home perspective
            team_matches.append({
                'date': row['date'],
                'team': row['home_team'],
                'opponent': row['away_team'],
                'goals_for': row['home_score'],
                'goals_against': row['away_score'],
                'is_home': 1,
                'outcome': row['outcome']
            })
            # Away perspective
            team_matches.append({
                'date': row['date'],
                'team': row['away_team'],
                'opponent': row['home_team'],
                'goals_for': row['away_score'],
                'goals_against': row['home_score'],
                'is_home': 0,
                'outcome': -row['outcome'] # Invert outcome for away team
            })
            
        flat_df = pd.DataFrame(team_matches)
        flat_df = flat_df.sort_values(['team', 'date'])
        
        # Calculate rolling averages per team
        for w in windows:
            flat_df[f'rolling_gf_{w}'] = flat_df.groupby('team')['goals_for'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            flat_df[f'rolling_ga_{w}'] = flat_df.groupby('team')['goals_against'].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
            flat_df[f'rolling_win_rate_{w}'] = flat_df.groupby('team')['outcome'].transform(lambda x: (x.shift(1) == 1).rolling(w, min_periods=1).mean())

        # Merge back to original df
        # This part is tricky: we need to join twice (once for home, once for away)
        home_stats = flat_df[flat_df['is_home'] == 1].drop(columns=['opponent', 'goals_for', 'goals_against', 'is_home', 'outcome'])
        away_stats = flat_df[flat_df['is_home'] == 0].drop(columns=['opponent', 'goals_for', 'goals_against', 'is_home', 'outcome'])
        
        df = df.merge(home_stats, left_on=['date', 'home_team'], right_on=['date', 'team'], how='left').drop(columns='team')
        df = df.merge(away_stats, left_on=['date', 'away_team'], right_on=['date', 'team'], how='left', suffixes=('_home', '_away')).drop(columns='team')
        
        return df

if __name__ == "__main__":
    from ingestion import ChronosIngestor
    
    ingestor = ChronosIngestor()
    ingestor.load_all()
    
    fe = FeatureEngineer()
    df = fe.generate_basic_features(ingestor.results)
    df = fe.generate_rolling_stats(df, windows=[5, 10])
    
    print("[+] Feature Engineering complete.")
    print(f"    - Columns: {df.columns.tolist()}")
    print(df[['date', 'home_team', 'away_team', 'rolling_win_rate_5_home', 'rolling_win_rate_5_away']].tail())
