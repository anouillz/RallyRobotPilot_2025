# dataset_npz.py
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from preprocess import Preprocessor, flip_image, flip_controls


class NpzDrivingDataset(Dataset):
    """
    Dataset pour fichiers .npz contenant :
      - images : (N, H, W, 3)
      - controls : (N, 4)

    Options :
      - augment_with_flip=True : ajoute images mirrored
      - temporal_shift=k       : X[i] -> Y[i+k]
    """

    def __init__(
        self,
        npz_paths,
        augment_with_flip=False,
        temporal_shift=0,
        preprocessor: Preprocessor | None = None
    ):
        super().__init__()

        # npz_paths peut être une string ou une liste
        if isinstance(npz_paths, str):
            self.files = sorted(glob.glob(npz_paths))
        else:
            self.files = list(npz_paths)

        if len(self.files) == 0:
            raise ValueError(f"Aucun fichier .npz trouvé (npz_paths={npz_paths})")

        print(f"Found {len(self.files)} .npz files.")

        self.augment_with_flip = augment_with_flip
        self.temporal_shift = temporal_shift

        # Préprocessor (si None → défaut)
        self.preprocessor = preprocessor if preprocessor is not None else Preprocessor()

        images_tensors = []
        controls_tensors = []

        # ---------- Chargement + prétraitement  ----------
        for file in self.files:
            data = np.load(file, allow_pickle=False)

            if "images" in data and "controls" in data:
                imgs = data["images"]
                ctrs = data["controls"]
            elif "features" in data and "labels" in data:
                imgs = data["features"]
                ctrs = data["labels"]
            else:
                raise ValueError(
                    f"{file} ne contient pas les clés attendues. Clés trouvées: {list(data.keys())}"
                )

            N = len(imgs)
            max_valid = N - self.temporal_shift  # nombre de samples utilisables

            for i in range(max_valid):
                img_np = imgs[i].astype(np.uint8)
                ctr_np = ctrs[i + self.temporal_shift].astype(np.float32)

                # Image PIL
                pil_img = Image.fromarray(img_np, mode="RGB")

                # Version normale
                img_t, ctr_t = self.preprocessor.process(pil_img, ctr_np)
                images_tensors.append(img_t)
                controls_tensors.append(ctr_t)

                # Version flipped si demandé
                if self.augment_with_flip:
                    pil_flip = flip_image(pil_img)
                    ctr_flip = flip_controls(ctr_np)
                    img_t_f, ctr_t_f = self.preprocessor.process(pil_flip, ctr_flip)
                    images_tensors.append(img_t_f)
                    controls_tensors.append(ctr_t_f)

        # Stack final
        self.images = torch.stack(images_tensors, dim=0)
        self.controls = torch.stack(controls_tensors, dim=0)

        print(
            f"Dataset prêt : {self.images.shape[0]} samples "
            f"(shift={self.temporal_shift}, flip={'Yes' if self.augment_with_flip else 'No'})."
        )

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.controls[idx]
