from PyQt6 import QtWidgets

import sys
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from scripts.data_collector import DataCollectionUI
from model import build_model
from config import DEVICE, IMAGE_RESIZED_DIMENSIONS

BEST_MODEL_PATH = "checkpoints/v9/best_model.pth"


class NNMsgProcessor:
    def __init__(self, threshold=0.5, debug=True, n_frames=3):
        self.debug = debug
        self.threshold = threshold
        self.n_frames = n_frames

        # Charger le modèle (3 * n_frames canaux)
        input_shape = (
            3 * n_frames,
            IMAGE_RESIZED_DIMENSIONS[1],
            IMAGE_RESIZED_DIMENSIONS[0],
        )
        self.model = build_model(input_shape).to(DEVICE)
        self.model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        # Préprocessing image (une seule frame)
        self.transform = T.Compose([
            T.Resize((IMAGE_RESIZED_DIMENSIONS[1], IMAGE_RESIZED_DIMENSIONS[0])),
            T.ToTensor(),  # -> (3, H, W), float32 [0,1]
        ])

        # Buffer des dernières frames (tensors 3xHxW)
        self.frame_buffer = []

        # États précédents : [F, B, L, R]
        self.prev_decisions = np.array([False, False, False, False], dtype=bool)
        self.commands = ["forward", "back", "left", "right"]

    def preprocess_image(self, message):
        """Retourne un tensor (3, H, W) pour une seule frame."""
        frame = getattr(message, "image", None)
        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        frame = frame.astype(np.uint8)
        image = Image.fromarray(frame)  # RGB
        tensor = self.transform(image)  # (3, H, W)
        return tensor

    def _make_input_tensor(self, frame_tensor: torch.Tensor) -> torch.Tensor | None:
        """
        Ajoute la frame au buffer et construit un tensor (1, 3*n_frames, H, W).
        Si on n'a pas encore assez de frames, on répète la première.
        """
        self.frame_buffer.append(frame_tensor)
        if len(self.frame_buffer) > self.n_frames:
            self.frame_buffer.pop(0)

        if len(self.frame_buffer) == 0:
            return None

        # Si pas encore assez de frames, on pad avec la première frame
        if len(self.frame_buffer) < self.n_frames:
            first = self.frame_buffer[0]
            num_missing = self.n_frames - len(self.frame_buffer)
            frames = [first] * num_missing + self.frame_buffer
        else:
            frames = self.frame_buffer

        # frames : liste de n_frames tensors (3, H, W)
        stacked = torch.cat(frames, dim=0)          # (3*n_frames, H, W)
        return stacked.unsqueeze(0).to(DEVICE)      # (1, 3*n_frames, H, W)

    def nn_infer(self, message):
        frame_tensor = self.preprocess_image(message)
        if frame_tensor is None:
            return None, None, []

        img_tensor = self._make_input_tensor(frame_tensor)
        if img_tensor is None:
            return None, None, []

        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = torch.sigmoid(logits)[0]

        probs_np = probs.cpu().numpy()
        decisions = probs_np > self.threshold

        to_send = []
        for i, cmd in enumerate(self.commands):
            if decisions[i] != self.prev_decisions[i]:
                to_send.append((cmd, bool(decisions[i])))

        self.prev_decisions = decisions.copy()

        if self.debug:
            print(
                f"probs={np.round(probs_np, 3)} "
                f"decisions={[bool(d) for d in decisions]} "
                f"to_send={to_send}"
            )

        return probs_np, decisions, to_send

    def process_message(self, message, data_collector):
        _, _, to_send = self.nn_infer(message)
        for command, start in to_send:
            data_collector.onCarControlled(command, start)


if __name__ == "__main__":
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = NNMsgProcessor(threshold=0.4, debug=True, n_frames=1)
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()