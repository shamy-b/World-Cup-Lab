import pandas as pd
import hashlib
import os
from typing import Optional

class ChronosIngestor:
    """
    Data ingestion engine for the Chronos Football Intelligence Platform.
    Handles loading, validation, and unique ID generation.
    """
    
    def __init__(self, data_dir: str = "dataset"):
        self.data_dir = data_dir
        self.results: Optional[pd.DataFrame] = None
        self.shootouts: Optional[pd.DataFrame] = None
        self.goalscorers: Optional[pd.DataFrame] = None
        self.former_names: Optional[pd.DataFrame] = None

    def load_all(self):
        """Loads all CSV files from the dataset directory."""
        print(f"[*] Ingesting data from {self.data_dir}...")
        
        self.results = pd.read_csv(os.path.join(self.data_dir, "results.csv"))
        self.shootouts = pd.read_csv(os.path.join(self.data_dir, "shootouts.csv"))
        self.goalscorers = pd.read_csv(os.path.join(self.data_dir, "goalscorers.csv"))
        self.former_names = pd.read_csv(os.path.join(self.data_dir, "former_names.csv"))
        
        self._validate_and_parse()
        self._generate_match_ids()
        
        print("[+] Ingestion complete.")
        print(f"    - Matches: {len(self.results)}")
        print(f"    - Shootouts: {len(self.shootouts)}")
        print(f"    - Goal events: {len(self.goalscorers)}")

    def _validate_and_parse(self):
        """Standardizes datatypes and validates core constraints."""
        # Drop matches with missing scores (future or cancelled)
        initial_count = len(self.results)
        self.results = self.results.dropna(subset=['home_score', 'away_score'])
        if len(self.results) < initial_count:
            print(f"[!] Dropped {initial_count - len(self.results)} matches with missing scores.")

        # Date parsing
        for df in [self.results, self.shootouts, self.goalscorers]:
            df['date'] = pd.to_datetime(df['date'])
            
        # Score validation
        if (self.results['home_score'] < 0).any() or (self.results['away_score'] < 0).any():
            print("[!] Warning: Negative scores detected. Cleaning...")
            self.results = self.results[(self.results['home_score'] >= 0) & (self.results['away_score'] >= 0)]

    def _generate_match_ids(self):
        """
        Generates unique Match IDs following the blueprint: YEAR_HOME_AWAY_HASH.
        We use a readable format: YYYYMMDD_Home_Away
        """
        def create_id(row):
            date_str = row['date'].strftime('%Y%m%d')
            home = str(row['home_team']).replace(" ", "")
            away = str(row['away_team']).replace(" ", "")
            return f"{date_str}_{home}_{away}"

        self.results['match_id'] = self.results.apply(create_id, axis=1)
        
        # Sync match_ids to other tables if possible (based on date/teams)
        # This is crucial for joining tables later
        print("[*] Generating unique Match IDs...")

    def get_summary(self):
        return {
            "matches": self.results.shape,
            "date_range": (self.results['date'].min(), self.results['date'].max())
        }

if __name__ == "__main__":
    ingestor = ChronosIngestor()
    ingestor.load_all()
    summary = ingestor.get_summary()
    print(f"[*] Timeline: {summary['date_range'][0].year} to {summary['date_range'][1].year}")
