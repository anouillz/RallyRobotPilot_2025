# preprocess.py
import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS


def flip_image(image: Image.Image) -> Image.Image:
    """Retourne l'image mirroir (flip horizontal)."""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def flip_controls(ctrl: np.ndarray) -> np.ndarray:
    """
    ctrl = [F, B, L, R]
    On inverse uniquement Left <-> Right.
    """
    F, B, L, R = ctrl
    return np.array([F, B, R, L], dtype=np.float32)


class Preprocessor:
    """
    Pipeline DÉTERMINISTE :
      - crop haut / bas
      - réduction (downscale)
      - resize final
      - normalisation
    AUCUN flip aléatoire ici.
    """

    def __init__(
        self,
        enable_crop_top: bool = False,
        enable_crop_bottom: bool = False,
        enable_downscale: bool = False,
        enable_resize: bool = False,
        enable_normalize: bool = False,
        crop_top_ratio: float = 0.35,
        crop_bottom_ratio: float = 0.15,
        downscale_factor: int = 2,
    ):
        self.enable_crop_top = enable_crop_top
        self.enable_crop_bottom = enable_crop_bottom
        self.enable_downscale = enable_downscale
        self.enable_resize = enable_resize
        self.enable_normalize = enable_normalize

        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.downscale_factor = downscale_factor

        self.resize_transform = T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH))
        self.normalize_transform = T.Normalize(
            mean=[0.5] * NUM_CHANNELS,
            std=[0.5] * NUM_CHANNELS,
        )

    # --------- CROP haut / bas ----------
    def crop_image(self, image: Image.Image) -> Image.Image:
        w, h = image.size

        y0 = int(h * self.crop_top_ratio) if self.enable_crop_top else 0
        y1 = h - int(h * self.crop_bottom_ratio) if self.enable_crop_bottom else h

        if y1 <= y0:
            y1 = y0 + 1

        return image.crop((0, y0, w, y1))

    # --------- DOWNSCALE ----------
    def downscale_image(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        new_w = max(1, w // self.downscale_factor)
        new_h = max(1, h // self.downscale_factor)
        return image.resize((new_w, new_h), Image.BILINEAR)

    # --------- PIPELINE complet ----------
    def process(self, image_pil: Image.Image, controls_np: np.ndarray):
        """
        Retourne (image_tensor, controls_tensor)
        Le flip éventuel est géré DEHORS (dans le dataset ou l'autopilot).
        """

        # Crop haut/bas
        if self.enable_crop_top or self.enable_crop_bottom:
            image_pil = self.crop_image(image_pil)

        # Downscale
        if self.enable_downscale:
            image_pil = self.downscale_image(image_pil)

        # Resize final
        if self.enable_resize:
            image_pil = self.resize_transform(image_pil)

        # ToTensor
        image_tensor = T.ToTensor()(image_pil)

        # Normalisation
        if self.enable_normalize:
            image_tensor = self.normalize_transform(image_tensor)

        controls_tensor = torch.from_numpy(controls_np.astype(np.float32))

        return image_tensor, controls_tensor
