import torch
import glob
import os

DATA_DIR = ""      # dossier où se trouvent tes .npz
NPZ_PATTERN = "*.npz"      # motif de recherche
CHECKPOINT_DIR = "checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")

def list_npz_files():
    pattern = os.path.join(DATA_DIR, NPZ_PATTERN)
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"Aucun fichier .npz trouvé avec le pattern {pattern}")
    return files

IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
NUM_CHANNELS = 3 

BATCH_SIZE = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4

NUM_OUTPUTS = 4

VAL_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 412

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRINT_EVERY = 50 
