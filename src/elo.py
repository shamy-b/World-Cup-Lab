import pandas as pd
import numpy as np

class EloSystem:
    """
    Dynamic Elo Rating System for Chronos.
    Includes tournament weighting, margin-of-victory adjustments,
    and proper tournament category mapping.
    """
    
    def __init__(self, initial_rating=1500):
        self.ratings = {}
        self.initial_rating = initial_rating
        
        # Comprehensive tournament tier mapping
        # Tier 1: Major finals (K=60)
        # Tier 2: Major qualifiers & continental finals (K=50)
        # Tier 3: Secondary continental & Nations League (K=40)
        # Tier 4: Minor tournaments (K=30)
        # Tier 5: Friendlies (K=20)
        self.tournament_tiers = {
            # Tier 1
            'FIFA World Cup': 60,
            'Confederations Cup': 55,
            # Tier 2
            'UEFA Euro': 50,
            'Copa América': 50,
            'African Cup of Nations': 50,
            'AFC Asian Cup': 50,
            'Gold Cup': 50,
            'FIFA World Cup qualification': 45,
            'UEFA Euro qualification': 45,
            # Tier 3
            'UEFA Nations League': 40,
            'CONMEBOL–UEFA Cup of Champions': 40,
            'African Cup of Nations qualification': 40,
            'AFC Asian Cup qualification': 40,
            'CONCACAF Nations League': 40,
            'Copa América qualification': 40,
            # Tier 4
            'Olympic Games': 35,
            'Gold Cup qualification': 35,
            'Gulf Cup': 30,
            'Baltic Cup': 30,
            'British Home Championship': 30,
            'CECAFA Cup': 30,
            'COSAFA Cup': 30,
            'AFF Championship': 30,
            'CFU Caribbean Cup': 30,
            'CFU Caribbean Cup qualification': 30,
            'SAFF Cup': 30,
            'Arab Cup': 30,
            'Asian Games': 30,
            'Pacific Games': 30,
            'Island Games': 25,
            # Tier 5
            'Friendly': 20,
        }

    def _get_tournament_k(self, tournament):
        """Returns the K-factor for a tournament. Falls back to keyword matching."""
        if tournament in self.tournament_tiers:
            return self.tournament_tiers[tournament]
        
        # Keyword-based fallback for tournaments not in the map
        t_lower = tournament.lower()
        if 'world cup' in t_lower:
            return 50
        elif 'qualification' in t_lower:
            return 40
        elif 'cup' in t_lower or 'championship' in t_lower or 'nations league' in t_lower:
            return 35
        elif 'games' in t_lower or 'olympic' in t_lower:
            return 30
        elif 'friendly' in t_lower:
            return 20
        else:
            return 30  # Default for unknown tournaments

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
        k = self._get_tournament_k(tournament)
        
        # Margin of Victory Multiplier
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
        
        return r_h, r_a

    def process_history(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates through match history and generates Elo features.
        """
        df = df.sort_values('date').reset_index(drop=True)
        elo_h_list = []
        elo_a_list = []
        
        print("[*] Processing Elo history...")
        for _, row in df.iterrows():
            old_h = self.get_rating(row['home_team'])
            old_a = self.get_rating(row['away_team'])
            
            elo_h_list.append(old_h)
            elo_a_list.append(old_a)
            
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
    
    top_teams = sorted(elo.ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Teams by Current Elo:")
    for i, (team, rating) in enumerate(top_teams, 1):
        print(f"{i}. {team}: {rating:.1f}")
