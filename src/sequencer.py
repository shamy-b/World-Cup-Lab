import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class FootballSequenceDataset(Dataset):
    """
    PyTorch Dataset for football match sequences.
    Converts historical matches into fixed-length windows.
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class TeamSequencer:
    """
    Prepares sequential data for LSTM/Transformer models.
    """
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def create_sequences(self, df, feature_cols):
        """
        Groups matches by team and creates rolling sequence windows.
        """
        # Flatten matches as we did in feature engineering
        team_matches = []
        for _, row in df.iterrows():
            common_data = {
                'date': row['date'],
                'tournament': row['tournament'],
                'outcome': row['outcome']
            }
            # Home
            h_data = common_data.copy()
            h_data.update({'team': row['home_team'], 'is_home': 1})
            for col in feature_cols:
                if col.endswith('_home'): h_data[col.replace('_home', '')] = row[col]
            team_matches.append(h_data)
            
            # Away
            a_data = common_data.copy()
            a_data.update({'team': row['away_team'], 'is_home': 0, 'outcome': -row['outcome']})
            for col in feature_cols:
                if col.endswith('_away'): a_data[col.replace('_away', '')] = row[col]
            team_matches.append(a_data)

        flat_df = pd.DataFrame(team_matches).sort_values(['team', 'date'])
        
        # Base features for the sequence (e.g., Elo, win rate, goals)
        base_features = [c.replace('_home', '') for c in feature_cols if c.endswith('_home')]
        
        all_sequences = []
        all_targets = []
        
        print(f"[*] Creating sequences (length={self.sequence_length})...")
        for team, group in flat_df.groupby('team'):
            if len(group) < self.sequence_length + 1:
                continue
                
            values = group[base_features].values
            outcomes = group['outcome'].values
            
            for i in range(len(group) - self.sequence_length):
                seq = values[i : i + self.sequence_length]
                # Target is the outcome of the NEXT match
                target = outcomes[i + self.sequence_length] + 1 # Map -1,0,1 to 0,1,2
                
                all_sequences.append(seq)
                all_targets.append(target)
                
        return np.array(all_sequences), np.array(all_targets)

if __name__ == "__main__":
    from baseline_model import BaselineModel
    
    baseline = BaselineModel()
    data, features = baseline.prepare_data()
    
    sequencer = TeamSequencer(sequence_length=5)
    X_seq, y_seq = sequencer.create_sequences(data, features)
    
    print(f"[+] Sequences generated: {X_seq.shape}")
    print(f"    - Target distribution: {np.bincount(y_seq)}")
