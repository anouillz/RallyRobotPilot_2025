# train.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn

from config import (
    list_npz_files,
    DEVICE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    VAL_SPLIT,
    SHUFFLE_DATASET,
    RANDOM_SEED,
    CHECKPOINT_DIR,
    BEST_MODEL_PATH,
    PRINT_EVERY,
)
from dataset_npz import NpzDrivingDataset
from model import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, criterion, optimizer, dataloader, epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    total_samples = 0
    count_batches = 0

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)  # (B, 4)

        optimizer.zero_grad()
        outputs = model(images)       # (B, 4)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        running_loss += loss.item()
        count_batches += 1

        if (batch_idx + 1) % PRINT_EVERY == 0:
            avg_loss = running_loss / count_batches
            print(f"[Epoch {epoch+1}][Batch {batch_idx+1}] loss: {avg_loss:.4f}")
            running_loss = 0.0
            count_batches = 0

    epoch_loss = total_loss / max(total_samples, 1)
    return epoch_loss


def evaluate(model, criterion, dataloader):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)


def main():
    set_seed(RANDOM_SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # optimisation cudnn pour CNN avec tailles fixes
    cudnn.benchmark = True

    print("DEVICE utilisé :", DEVICE)
    print("torch.cuda.is_available() :", torch.cuda.is_available())

    # Récupération des fichiers .npz
    npz_files = list_npz_files()
    print(f"Fichiers de données trouvés ({len(npz_files)}) :")
    for f in npz_files:
        print("  -", f)

    # Un SEUL dataset global, prétraité offline (avec flip pour augmenter le train)
    full_dataset = NpzDrivingDataset(npz_files, augment_with_flip=True)

    dataset_size = len(full_dataset)
    val_size = int(VAL_SPLIT * dataset_size)
    train_size = dataset_size - val_size

    # Split train / val
    if SHUFFLE_DATASET:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED),
        )
    else:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
        )

    # DataLoaders – optimisés pour CUDA
    common_loader_kwargs = dict(
        num_workers=2,                
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        **common_loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **common_loader_kwargs,
    )

    # Modèle + optim
    model = build_model().to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    epochs_list = list(range(1, NUM_EPOCHS + 1))

    for epoch in range(NUM_EPOCHS):
        print(f"===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, epoch)
        val_loss = evaluate(model, criterion, val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train loss: {train_loss:.4f}")
        print(f"Validation loss: {val_loss:.4f}")

        # Sauvegarde si meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f">>> Nouveau meilleur modèle sauvegardé : {BEST_MODEL_PATH}")

    print("Entraînement terminé.")
    print(f"Meilleur loss de validation : {best_val_loss:.4f}")

    # --------- Tracer le graphique des pertes ---------
    plt.figure()
    plt.plot(epochs_list, train_losses, label="Train loss")
    plt.plot(epochs_list, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    loss_plot_path = os.path.join(CHECKPOINT_DIR, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Graphique des pertes sauvegardé dans : {loss_plot_path}")


if __name__ == "__main__":
    main()
