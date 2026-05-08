import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sequencer import TeamSequencer, FootballSequenceDataset
from baseline_model import BaselineModel

class FootballLSTM(nn.Module):
    """
    Research-grade LSTM for Latent Football Knowledge Extraction.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=3, dropout=0.2):
        super(FootballLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Use last time step
        return out

def train_dl_model():
    print("[*] Initializing Deep Learning Training Pipeline...")
    
    # 1. Prepare Data
    baseline = BaselineModel()
    data, features = baseline.prepare_data()
    
    sequencer = TeamSequencer(sequence_length=10)
    X, y = sequencer.create_sequences(data, features)
    
    dataset = FootballSequenceDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 2. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    
    model = FootballLSTM(input_dim=X.shape[2]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    # 4. Save
    torch.save(model.state_dict(), "models/football_lstm.pth")
    print("[*] Deep Learning model saved to models/football_lstm.pth")

if __name__ == "__main__":
    train_dl_model()
