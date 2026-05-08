import pandas as pd
import os

class TeamNormalizer:
    """
    Historical Normalization Engine.
    Manages team name transitions and political boundary changes.
    """
    
    def __init__(self, former_names_path: str = "dataset/former_names.csv"):
        self.former_names_df = pd.read_csv(former_names_path)
        self.former_names_df['start_date'] = pd.to_datetime(self.former_names_df['start_date'])
        self.former_names_df['end_date'] = pd.to_datetime(self.former_names_df['end_date'])
        
        # Create mapping dictionary: former -> current
        self.name_map = dict(zip(self.former_names_df['former'], self.former_names_df['current']))

    def normalize_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps all historical names to their current 'Successor' names.
        This is useful for long-term team embeddings and Elo consistency.
        """
        df = results_df.copy()
        
        print("[*] Normalizing historical team names...")
        
        # Simple mapping (ignoring date for now as the names are usually unique)
        df['home_team_normalized'] = df['home_team'].replace(self.name_map)
        df['away_team_normalized'] = df['away_team'].replace(self.name_map)
        
        changes = (df['home_team'] != df['home_team_normalized']).sum() + \
                  (df['away_team'] != df['away_team_normalized']).sum()
        
        print(f"[+] Normalization applied: {changes} name conversions made.")
        return df

    def get_lineage(self, current_name: str):
        """Returns all former names for a given modern team."""
        return self.former_names_df[self.former_names_df['current'] == current_name]['former'].tolist()

if __name__ == "__main__":
    # Test normalization
    from ingestion import ChronosIngestor
    
    ingestor = ChronosIngestor()
    ingestor.load_all()
    
    normalizer = TeamNormalizer()
    normalized_results = normalizer.normalize_results(ingestor.results)
    
    # Check a known change
    sample = normalized_results[normalized_results['home_team'] == 'Dahomey'].head(1)
    if not sample.empty:
        print(f"[*] Verified: Dahomey -> {sample['home_team_normalized'].values[0]}")
