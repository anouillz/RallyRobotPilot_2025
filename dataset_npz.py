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
      - images : (N, H, W, 3) uint8
      - controls : (N, 4) (F,B,L,R)

    Le prétraitement (crop, downscale, resize, normalisation) est
    fait UNE SEULE FOIS au chargement, puis stocké en tensors.

    Si augment_with_flip=True :
      -> pour chaque sample, on ajoute la version mirroir (flip horizontal)
         avec commandes L/R inversées.
    """

    def __init__(self, npz_paths, augment_with_flip: bool = True):
        super().__init__()

        # npz_paths peut être une string (pattern) ou une liste
        if isinstance(npz_paths, str):
            self.files = sorted(glob.glob(npz_paths))
        else:
            self.files = list(npz_paths)

        if len(self.files) == 0:
            raise ValueError(f"Aucun fichier .npz trouvé (npz_paths={npz_paths})")

        print(f"Found {len(self.files)} .npz files.")

        self.augment_with_flip = augment_with_flip

        # Preprocessor DÉTERMINISTE
        self.preprocessor = Preprocessor()

        images_tensors = []
        controls_tensors = []

        # ---------- Chargement + prétraitement OFFLINE ----------
        for file in self.files:
            data = np.load(file, allow_pickle=False)

            # Clés possibles dans le .npz
            if "images" in data and "controls" in data:
                imgs = data["images"]
                ctrs = data["controls"]
            elif "features" in data and "labels" in data:
                imgs = data["features"]
                ctrs = data["labels"]
            else:
                raise ValueError(
                    f"{file} ne contient pas 'images'/'controls' "
                    f"ni 'features'/'labels'. Clés: {list(data.keys())}"
                )

            if len(imgs) != len(ctrs):
                raise ValueError(f"Incohérence images/controls dans {file}")

            for i in range(len(imgs)):
                img_np = imgs[i].astype(np.uint8)       # (H, W, 3)
                ctr_np = ctrs[i].astype(np.float32)     # (4,)

                pil_img = Image.fromarray(img_np, mode="RGB")

                # Version normale
                img_t, ctr_t = self.preprocessor.process(pil_img, ctr_np)
                images_tensors.append(img_t)
                controls_tensors.append(ctr_t)

                # Version mirroir (si demandé)
                if self.augment_with_flip:
                    pil_flip = flip_image(pil_img)
                    ctr_flip = flip_controls(ctr_np)
                    img_t_f, ctr_t_f = self.preprocessor.process(pil_flip, ctr_flip)
                    images_tensors.append(img_t_f)
                    controls_tensors.append(ctr_t_f)

        self.images = torch.stack(images_tensors, dim=0)       # (N_total, C, H, W)
        self.controls = torch.stack(controls_tensors, dim=0)   # (N_total, 4)

        print(
            f"Dataset prêt : {self.images.shape[0]} samples "
            f"({'avec' if self.augment_with_flip else 'sans'} flip)."
        )

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.controls[idx]
