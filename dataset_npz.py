import numpy as np
import pickle
import torch
import lzma

from pathlib import Path
from torch.utils.data import Dataset
from preprocess import preprocess_image
import cv2


class DrivingDataset(Dataset):
    def __init__(self, dir_path: str, n_frames: int = 1, shift: int = 0, augment_flip=False):
        dir_path = Path(dir_path)
        all_images = []
        all_controls = []

        self.augment_flip = augment_flip

        for filepath in sorted(dir_path.glob("*.npz")):
            with lzma.open(filepath, "rb") as file:
                print(f"Loading {filepath}...")
                data = pickle.load(file)
            
            imgs = []
            ctrls = []
            for e in data:
                imgs.append(preprocess_image(e.image, to_tensor=False))  # (H,W,3)
                ctrls.append(e.current_controls)  # [F,B,L,R]
        
            imgs = np.array(imgs)
            ctrls = np.array(ctrls, dtype=np.float32)
            L = len(imgs)

            for t in range(n_frames - 1, L - shift):
                window = imgs[t - n_frames + 1 : t + 1]
                stacked = window.reshape(-1, *window.shape[2:])
                target = ctrls[t + shift]

                # --- entrée normale ---
                all_images.append(stacked)
                all_controls.append(target)

                # --- miroir horizontal (augmentation) ---
                if self.augment_flip:
                    flipped_imgs = np.array([cv2.flip(frame, 1) for frame in window])
                    flipped_stacked = flipped_imgs.reshape(-1, *flipped_imgs.shape[2:])

                    flipped_target = target.copy()
                    # swap left <-> right
                    flipped_target[2], flipped_target[3] = target[3], target[2]

                    all_images.append(flipped_stacked)
                    all_controls.append(flipped_target)

        self.images = all_images
        self.controls = all_controls

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).float()
        ctrl = torch.from_numpy(self.controls[idx]).float()
        return img, ctrl
