import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os

# Paths
PREPROCESSED_DATA_PATH = "SIGNLANG/datasets/preprocessed/"
MODEL_SAVE_PATH = "SIGNLANG/models/sign_model.pth"

# Custom Dataset
class SignDataset(Dataset):
    def __init__(self):
        self.data = [f for f in os.listdir(PREPROCESSED_DATA_PATH)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        file = self.data[idx]
        frames = np.load(os.path.join(PREPROCESSED_DATA_PATH, file))
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(int(file.split('_')[0]))  # Modify target as needed

# Define the Model (CNN+LSTM)
class SignLanguageModel(nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.lstm = nn.LSTM(50176, 128, batch_first=True)
        self.fc = nn.Linear(128, 100)  # Modify based on class count
    
    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        cnn_out = self.cnn(x.view(batch_size * timesteps, C, H, W))
        cnn_out = cnn_out.view(batch_size, timesteps, -1)
        lstm_out, _ = self.lstm(cnn_out)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Training Loop
def train_model():
    dataset = SignDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SignLanguageModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(10):  # Adjust epochs as necessary
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
