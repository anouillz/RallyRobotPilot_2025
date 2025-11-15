# preprocess.py
import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS


def flip_image(image):
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def flip_controls(ctrl):
    F, B, L, R = ctrl
    return np.array([F, B, R, L], dtype=np.float32)


class Preprocessor:
    def __init__(self,
                 use_augmentation=True,
                 crop_top_ratio=0.10,   # % du haut à retirer
                 crop_bottom_ratio=0.20, # % du bas à retirer
                 downscale_factor=2):    # réduction avant resize
        """
        crop_top_ratio    : fraction du haut à supprimer (0.0 à 0.9)
        crop_bottom_ratio : fraction du bas à supprimer (0.0 à 0.9)
        """
        self.use_aug = use_augmentation
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.downscale_factor = downscale_factor

        self.final_transform = T.Compose([
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * NUM_CHANNELS, std=[0.5] * NUM_CHANNELS),
        ])

    def reduce_image(self, image):
        """
        image : PIL.Image

        1) Crop haut
        2) Crop bas
        3) Downscale (réduction)
        """
        w, h = image.size

        # ----- 1) Crop haut -----
        crop_top = int(h * self.crop_top_ratio)

        # ----- 2) Crop bas -----
        crop_bottom = int(h * self.crop_bottom_ratio)

        # zone gardée : de crop_top jusqu'à h - crop_bottom
        y0 = crop_top
        y1 = h - crop_bottom

        if y1 <= y0:  
            # on évite un crop invalide
            y1 = y0 + 1

        image = image.crop((0, y0, w, y1))

        # ----- 3) Downscale -----
        cropped_h = y1 - y0
        new_w = max(1, w // self.downscale_factor)
        new_h = max(1, cropped_h // self.downscale_factor)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        return image

    def process(self, image_pil, controls_np):
        """
        Pipeline complet :
        - Crop haut + bas
        - Downscale
        - Optionnel : Flip horizontal (avec inversion L/R)
        - Resize final
        - Normalisation
        """

        # --- Crop + downscale ---
        image_pil = self.reduce_image(image_pil)

        # --- Data augmentation (flip) ---
        if self.use_aug and np.random.rand() < 0.5:
            image_pil = flip_image(image_pil)
            controls_np = flip_controls(controls_np)

        # --- Resize + normalisation ---
        image_tensor = self.final_transform(image_pil)
        controls_tensor = torch.from_numpy(controls_np.astype(np.float32))

        return image_tensor, controls_tensor
