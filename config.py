# config.py
import os
import glob
import torch

# --- chemins ---
DATA_DIR = ""          # dossier où sont les .npz
NPZ_PATTERN = "record_*.npz"   # motif pour les fichiers

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# --- découverte des fichiers npz ---
def list_npz_files():
    pattern = os.path.join(DATA_DIR, NPZ_PATTERN)
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"Aucun fichier .npz trouvé avec le pattern {pattern}")
    return files

# --- données / images ---
IMAGE_RESIZED_DIMENSIONS = (128, 128)
NUM_CHANNELS = 3      # N&B
SEQ_LEN = 1
NUM_OUTPUTS = 4       # [Forward, Back, Left, Right]


# --- apprentissage ---
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4

VAL_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 342

# device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# logs
PRINT_EVERY = 50  # fréquence d'affichage des logs d'entraînement
