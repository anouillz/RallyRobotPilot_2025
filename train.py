# train.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, random_split
import torch.backends.cudnn as cudnn
from contextlib import nullcontext

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
from dataset_npz import DrivingDataset
from model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, criterion, optimizer, dataloader, epoch, scaler=None):
    """
    Entraîne le modèle sur une epoch.
    Si scaler n'est pas None -> utilise AMP (mixed precision).
    """
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    total_samples = 0
    count_batches = 0

    use_amp = False

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)  # (B, 4)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(images)       # (B, 4)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
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
    """
    Évalue le modèle sur le dataloader (validation).
    Utilise AMP si on est en CUDA, pour aller plus vite.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    use_amp = False

    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with amp_ctx():
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / max(total_samples, 1)

def compute_confusion_matrix(model, dataloader, threshold=0.5):
    """
    Calcule une 'confusion matrix' par action (multi-label binaire).
    On retourne un array (4, 4) :
        lignes  = [forward, back, left, right]
        colonnes = [TN, FP, FN, TP]
    """
    model.eval()
    num_actions = 4
    cm = np.zeros((num_actions, 4), dtype=np.int64)  # TN, FP, FN, TP

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)  # (B, 4)

            logits = model(images)
            probs = torch.sigmoid(logits)

            preds = (probs > threshold).cpu().numpy().astype(int)   # (B,4)
            targs = targets.cpu().numpy().astype(int)               # (B,4)

            for j in range(num_actions):
                p = preds[:, j]
                t = targs[:, j]

                tn = np.sum((p == 0) & (t == 0))
                fp = np.sum((p == 1) & (t == 0))
                fn = np.sum((p == 0) & (t == 1))
                tp = np.sum((p == 1) & (t == 1))

                cm[j, 0] += tn
                cm[j, 1] += fp
                cm[j, 2] += fn
                cm[j, 3] += tp

    return cm


def plot_confusion_matrix(cm, actions, save_path):
    """
    Trace et sauvegarde la matrice de confusion multi-label sous forme de heatmap.
    cm : (4,4)  (TN, FP, FN, TP par action)
    actions : liste de noms d'actions
    """
    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(actions)))
    ax.set_xticklabels(["TN", "FP", "FN", "TP"])
    ax.set_yticklabels(actions)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Annoter chaque case avec la valeur
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center")

    ax.set_title("Matrice de confusion (multi-label)")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)



def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Optimisation cuDNN pour CNN avec tailles d'images fixes
    cudnn.benchmark = True

    print("DEVICE utilisé :", DEVICE)
    print("torch.cuda.is_available() :", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU :", torch.cuda.get_device_name(0))

    # Récupération des fichiers .npz
    path_files = "./"

    # Dataset global, prétraité offline (avec flip si tu veux l'augmentation)
    full_dataset = DrivingDataset(path_files ,augment_flip=True)

    val_size = int(VAL_SPLIT * len(full_dataset))
    train_size = len(full_dataset) - val_size

    # Split train / val
    if SHUFFLE_DATASET:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator(),
        )
    else:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
        )

    print(f"Total samples: {len(full_dataset)} (train={len(train_dataset)}, val={len(val_dataset)})")

    cpu_count = os.cpu_count() or 4
    num_workers = min(8, max(1, cpu_count - 1))
    print(f"Using num_workers = {num_workers}")

    common_loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=(DEVICE.startswith("cuda")),
        persistent_workers=(num_workers > 0),
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
    x0, y0 = full_dataset[0]

    C, H, W = x0.shape

    model =build_model((C, H, W)).to(DEVICE)
    # Modèle + optim
    pos_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], device=DEVICE)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # AMP scaler (uniquement si CUDA)
    scaler = None
    if scaler is not None:
        print("AMP (mixed precision) ACTIVÉ.")
    else:
        print("AMP désactivé (CPU ou pas de CUDA).")

    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    epochs_list = list(range(1, NUM_EPOCHS + 1))

    for epoch in range(NUM_EPOCHS):
        print(f"===== Epoch {epoch+1}/{NUM_EPOCHS} =====")
        train_loss = train_one_epoch(model, criterion, optimizer, train_loader, epoch, scaler=scaler)
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
    print("Calcul de la matrice de confusion sur le set de validation...")

    # On recharge le meilleur modèle pour être cohérent
    best_model = build_model((C, H, W)).to(DEVICE)
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    cm = compute_confusion_matrix(best_model, val_loader, threshold=0.5)
    actions = ["forward", "back", "left", "right"]

    print("Matrice de confusion (TN, FP, FN, TP) par action :")
    for i, act in enumerate(actions):
        tn, fp, fn, tp = cm[i]
        print(f"{act:8s} -> TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    cm_plot_path = os.path.join(CHECKPOINT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(cm, actions, cm_plot_path)
    print(f"Matrice de confusion sauvegardée dans : {cm_plot_path}")



if __name__ == "__main__":
    main()
