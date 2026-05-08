"""
Train a simple MLP on MNIST and export to binary file for C inference.
"""
import os
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MLP, ModelArgs

# -----------------------------------------------------------------------------
# Hyperparameters
batch_size = 64
learning_rate = 1e-3
epochs = 5
eval_interval = 100   # print loss every N batches
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = 'out'
model_bin_name = 'model.bin'

# -----------------------------------------------------------------------------
# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean, std
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------------------------------------------------------
# Model, optimizer, loss
args = ModelArgs()
model = MLP(args).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# -----------------------------------------------------------------------------
# Training
os.makedirs(out_dir, exist_ok=True)
model.train()

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)  # flatten to 784

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % eval_interval == 0:
            print(f"Epoch {epoch} [{batch_idx:4d}/{len(train_loader)}]  Loss: {loss.item():.4f}")

        # Quick evaluation on test set every 10*eval_interval batches
        if batch_idx % (eval_interval * 10) == 0:
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    data = data.view(data.size(0), -1)
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
            test_loss /= len(test_loader.dataset)
            acc = 100. * correct / len(test_loader.dataset)
            print(f"  Test  ----  avg loss: {test_loss:.4f}, accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")
            model.train()

# Save checkpoint and binary export
torch.save({'model_state_dict': model.state_dict(), 'args': args},
           os.path.join(out_dir, 'checkpoint.pt'))
