import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 

import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from scripts.model_c1.cnn_lstm import CNNLSTM

TRAIN_PATH = "scripts/dataset/all_train_seq5.npz"
VAL_PATH = "scripts/dataset/all_val_seq5.npz"

class DrivingDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"]          # (N, seq, H, W, 3)
        self.Y = data["Y"]          # (N, 4)

        # convert to torch-friendly format
        self.X = self.X.astype(np.float32) / 255.0  # normalize
        self.Y = self.Y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),    # (seq, H, W, 3)
            torch.from_numpy(self.Y[idx])     # (4,)
        )


# trraining 
def train_model(
    train_npz=TRAIN_PATH,
    val_npz=VAL_PATH,
    batch_size=16,
    lr=1e-4,
    num_epochs=20,
    device=None
):

    # device (if calypso, use GPU)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # Datasets
    train_set = DrivingDataset(train_npz)
    val_set = DrivingDataset(val_npz)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Model
    model = CNNLSTM(num_actions=4).to(device)

    # Loss (multi-label)
    criterion = nn.BCELoss()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # For saving best model
    best_val_loss = float("inf")

    # Lists to store losses for plotting
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for X, Y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()

            outputs = model(X)   # (B, 4)
            loss = criterion(outputs, Y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X.size(0)

        epoch_train_loss = running_loss / len(train_set)

        # validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for X, Y in val_loader:
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)
                loss = criterion(outputs, Y)
                val_loss += loss.item() * X.size(0)

        epoch_val_loss = val_loss / len(val_set)

        # store for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {epoch_train_loss:.4f}")
        print(f"Val   Loss: {epoch_val_loss:.4f}")

        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_cnn_lstm.pth")
            print("Saved best model so far")

    # plot train / val losses
    epochs = range(1, num_epochs + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNN+LSTM training / validation loss")
    plt.grid(True)
    plt.legend()
    #plt.savefig("loss_curve.png", bbox_inches="tight")
    plt.show() 


    print("\nTraining finished!")
    print("Best model saved as: best_cnn_lstm.pth")
    print("Loss curve saved to loss_curve.png")


if __name__ == "__main__":
    train_model(
        train_npz=TRAIN_PATH,
        val_npz=VAL_PATH,
        batch_size=16,
        lr=1e-4,
        num_epochs=20
    )
