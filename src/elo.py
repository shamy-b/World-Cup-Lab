import pandas as pd
import numpy as np

class EloSystem:
    """
    Dynamic Elo Rating System for Chronos.
    Includes tournament weighting and margin-of-victory adjustments.
    """
    
    def __init__(self, initial_rating=1500, k_factor=32):
        self.ratings = {} # team_name -> current_rating
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        
        # Tournament weights as per blueprint
        self.tournament_weights = {
            'FIFA World Cup': 60,
            'Continental championship': 50,
            'FIFA World Cup qualification': 40,
            'Nations League': 30,
            'Friendly': 20
        }

    def get_rating(self, team):
        return self.ratings.get(team, self.initial_rating)

    def compute_expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, home_team, away_team, home_score, away_score, tournament):
        r_h = self.get_rating(home_team)
        r_a = self.get_rating(away_team)
        
        # Actual score (1 for win, 0.5 for draw, 0 for loss)
        if home_score > away_score:
            s_h, s_a = 1.0, 0.0
        elif home_score < away_score:
            s_h, s_a = 0.0, 1.0
        else:
            s_h, s_a = 0.5, 0.5
            
        e_h = self.compute_expected_score(r_h, r_a)
        e_a = 1 - e_h
        
        # Tournament Weight (K)
        # Default to 20 if tournament not in map
        k = self.tournament_weights.get(tournament, 20)
        
        # Margin of Victory Multiplier (Blueprint improvement)
        # Based on World Football Elo Ratings formula
        diff = abs(home_score - away_score)
        if diff <= 1:
            g = 1
        elif diff == 2:
            g = 1.5
        else:
            g = (11 + diff) / 8
            
        # Update
        new_r_h = r_h + k * g * (s_h - e_h)
        new_r_a = r_a + k * g * (s_a - e_a)
        
        self.ratings[home_team] = new_r_h
        self.ratings[away_team] = new_r_a
        
        return r_h, r_a # Return old ratings for match-day features

    def process_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through match history and generates Elo features.
        """
        df = df.sort_values('date')
        elo_h_list = []
        elo_a_list = []
        
        print("[*] Processing Elo history...")
        for _, row in df.iterrows():
            # Get ratings BEFORE the match
            old_h = self.get_rating(row['home_team'])
            old_a = self.get_rating(row['away_team'])
            
            elo_h_list.append(old_h)
            elo_a_list.append(old_a)
            
            # Update ratings AFTER the match
            self.update_ratings(
                row['home_team'], row['away_team'],
                row['home_score'], row['away_score'],
                row['tournament']
            )
            
        df['elo_home'] = elo_h_list
        df['elo_away'] = elo_a_list
        df['elo_diff'] = df['elo_home'] - df['elo_away']
        
        print("[+] Elo ratings generated.")
        return df

if __name__ == "__main__":
    from ingestion import ChronosIngestor
    
    ingestor = ChronosIngestor()
    ingestor.load_all()
    
    elo = EloSystem()
    df = elo.process_history(ingestor.results)
    
    # Show top teams
    top_teams = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Teams by Current Elo:")
    for i, (team, rating) in enumerate(top_teams, 1):
        print(f"{i}. {team}: {rating:.1f}")
