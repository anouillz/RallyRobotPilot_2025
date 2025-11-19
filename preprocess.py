import numpy as np
import torch
import cv2

from config import IMAGE_RESIZED_DIMENSIONS


def preprocess_image(img: np.ndarray,
                         size=IMAGE_RESIZED_DIMENSIONS,
                         to_tensor: bool = True) -> torch.Tensor | np.ndarray:
    w, h = size
    if (h, w) != (128, 128):
        img = cv2.resize(img, (w, h)).astype(np.float32)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))

    if to_tensor:
        return torch.from_numpy(img)
    return img