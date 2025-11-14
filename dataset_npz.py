import lzma
import pickle
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from preprocess import Preprocessor


class NpzDrivingDataset(Dataset):
    """
    Dataset pour fichiers 'record_*.npz' qui sont en réalité :
      - des fichiers XZ compressés
      - contenant un pickle Python (par ex. une liste de SensingSnapshot)

    Pour chaque snapshot, on extrait :
      - snapshot.image  -> np.array (H, W, 3) uint8
      - snapshot.current_controls -> (F, B, L, R)

    Les commandes sont converties en vecteur float32 de taille 4, normalisé en 0/1.
    On applique :
      - Resize
      - Normalisation
      - (optionnel) Flip horizontal + inversion L/R
    """

    def __init__(self, npz_paths, use_augmentation=True):
        super().__init__()
        self.npz_paths = list(npz_paths)
        if len(self.npz_paths) == 0:
            raise ValueError("Aucun fichier de données fourni au dataset.")

        self.data_files = [self._load_pickle_xz(p) for p in self.npz_paths]

        self.index = []
        for file_idx, snapshots in enumerate(self.data_files):
            if not hasattr(snapshots, "__len__"):
                raise ValueError(
                    f"Le contenu de {self.npz_paths[file_idx]} n'est pas une liste/sequence."
                )

            for sample_idx, snap in enumerate(snapshots):
                # on ne garde que les snapshots qui ont une image
                img = getattr(snap, "image", None)
                if img is None:
                    continue
                self.index.append((file_idx, sample_idx))

        if len(self.index) == 0:
            raise ValueError("Aucun snapshot avec image trouvée dans les fichiers fournis.")

        # Préprocesseur (resize, normalization, flip, etc.)
        self.preprocessor = Preprocessor(use_augmentation=use_augmentation)

    # ------------------------------------------------------------
    # Chargement du fichier : XZ + pickle
    # ------------------------------------------------------------
    def _load_pickle_xz(self, path):
        """
        Ouvre un fichier compresse XZ et charge son contenu via pickle.
        On suppose que le module d'origine (rallyrobopilot.*) existe
        sur ta machine, donc pickle.load fonctionnera.
        """
        with lzma.open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    # ------------------------------------------------------------
    # Méthodes standard Dataset
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index[idx]

        snapshots = self.data_files[file_idx]
        snap = snapshots[sample_idx]

        # Récupération de l'image
        img_np = np.array(snap.image, dtype=np.uint8)  # (H, W, 3)
        image_pil = Image.fromarray(img_np, mode="RGB")

        # Récupération des commandes
        controls = getattr(snap, "current_controls", (0, 0, 0, 0))
        # Convertir en array float32 [F,B,L,R]
        ctrl_np = np.array(controls, dtype=np.float32)

        # Si ce n'est pas déjà 0/1, on force (>0 -> 1)
        ctrl_np = (ctrl_np > 0.5).astype(np.float32)

        # Preprocessing + (éventuellement) augmentation
        image_tensor, controls_tensor = self.preprocessor.process(image_pil, ctrl_np)

        return image_tensor, controls_tensor
