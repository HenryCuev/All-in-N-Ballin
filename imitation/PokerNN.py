import pandas as pd

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("poker_training_data.csv")

X = np.stack(df["hole_cards_encoded"].values)


y = df["action_label"].values

X = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in X], dtype=np.float32)


X = X[:,:16]

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Define a simple feedforward neural network
class PokerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PokerNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

input_size = X.shape[1]


model = PokerNN(16, 128, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_tensor)
    loss = criterion(output, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "poker_model.pth")
print("Training Complete!")
