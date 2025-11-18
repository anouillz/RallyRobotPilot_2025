from mpi4py import MPI
import os, sys, json, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT)

from scripts.model_c1.cnn_lstm import CNNLSTM

TRAIN_PATH = "scripts/dataset/all_train_seq5.npz"
VAL_PATH   = "scripts/dataset/all_val_seq5.npz"

CONFIG_DIR = "scripts/hparam_configs"
RESULTS_DIR = "scripts/hparam_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# mpi init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# dataset class
class DrivingDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = data["X"].astype(np.float32) / 255.0
        self.Y = data["Y"].astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X[idx]),
            torch.from_numpy(self.Y[idx])
        )

# load all config files
if rank == 0:
    configs = sorted([
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR)
        if f.endswith(".json")
    ])
else:
    configs = None

configs = comm.bcast(configs, root=0)

# Assign config to each rank
if rank >= len(configs):
    print(f"[Rank {rank}] No config assigned → exiting.")
    MPI.Finalize()
    sys.exit()

config_path = configs[rank]
with open(config_path, "r") as f:
    cfg = json.load(f)

print(f"[Rank {rank}] Using config {config_path}")

# load datasets
train_set = DrivingDataset(TRAIN_PATH)
val_set   = DrivingDataset(VAL_PATH)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=16, shuffle=False)

# Build model
model = CNNLSTM(
    num_actions=4,
    hidden_size=cfg["hidden_size"],
    dropout=cfg["dropout"]
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])

# training
NUM_EPOCHS = 30
best_val_loss = float("inf")

for epoch in range(NUM_EPOCHS):

    model.train()
    train_loss = 0

    for X, Y in tqdm(train_loader, desc=f"[Rank {rank}] Epoch {epoch+1}/{NUM_EPOCHS}"):
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)

    train_loss /= len(train_set)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            loss = criterion(outputs, Y)
            val_loss += loss.item() * X.size(0)

    val_loss /= len(val_set)

    print(f"[Rank {rank}] Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"{RESULTS_DIR}/model_rank_{rank}.pth")

# save results
result = {
    "rank": rank,
    "config": cfg,
    "best_val_loss": best_val_loss
}

with open(f"{RESULTS_DIR}/result_rank_{rank}.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"[Rank {rank}] Finished. Best val loss = {best_val_loss:.4f}")
