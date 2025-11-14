import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image

from config import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS


# ----- Transformations de base (resize + normalisation) -----
def base_transform():
    return T.Compose([
        T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.5] * NUM_CHANNELS, std=[0.5] * NUM_CHANNELS),
    ])


# ----- Flip horizontal -----
def flip_image(image):
    """image : PIL.Image"""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def flip_controls(ctrl):
    """
    ctrl = [F, B, L, R]
    On inverse L et R uniquement.
    """
    F, B, L, R = ctrl
    return np.array([F, B, R, L], dtype=np.float32)


# ----- Pipeline preprocessing complet -----
class Preprocessor:
    def __init__(self, use_augmentation=True):
        self.use_aug = use_augmentation
        self.transform = base_transform()

    def process(self, image_pil, controls_np):
        """
        image_pil : PIL.Image
        controls_np : np.array([F,B,L,R])
        """
        # --- Augmentation : flip horizontal ---
        if self.use_aug and np.random.rand() < 0.5:
            image_pil = flip_image(image_pil)
            controls_np = flip_controls(controls_np)

        # --- Transformations de base ---
        image_tensor = self.transform(image_pil)
        controls_tensor = torch.from_numpy(controls_np.astype(np.float32))

        return image_tensor, controls_tensor
