from PyQt6 import QtWidgets

import sys
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from scripts.data_collector import DataCollectionUI
from model import build_model
from config import  DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS
BEST_MODEL_PATH = "checkpoints/v5 modelv2 30 epoch/best_model.pth"

class NNMsgProcessor:
    def __init__(self, threshold=0.3, debug=True):
        self.debug = debug
        self.threshold = threshold

        # Chargement du modèle
        self.model = build_model().to(DEVICE)
        self.model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        self.model.eval()

        # Même preprocessing que pour l'entraînement (sans augmentation)
        self.transform = T.Compose([
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=[0.5] * NUM_CHANNELS, std=[0.5] * NUM_CHANNELS),
        ])

        # État précédent des décisions [F, B, L, R]
        self.prev_decisions = np.array([False, False, False, False], dtype=bool)

        # Ordre des commandes correspondant aux sorties du réseau
        self.commands = ["forward", "back", "left", "right"]

    def preprocess_image(self, message):
        """
        message : snapshot avec un attribut .image
        .image doit être un np.array (H, W, 3) uint8 (RGB)
        """
        frame = getattr(message, "image", None)
        if frame is None:
            return None

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)

        frame = frame.astype(np.uint8)
        image = Image.fromarray(frame, mode="RGB")

        tensor = self.transform(image).unsqueeze(0).to(DEVICE)  # (1, C, H, W)
        return tensor

    def nn_infer(self, message):
        """
        Renvoie:
          probs     : np.array shape (4,) float32 in [0,1]
          decisions : np.array shape (4,) bool
          to_send   : list[(command:str, bool)]
        """
        img_tensor = self.preprocess_image(message)
        if img_tensor is None:
            # Pas de nouvelle commande si pas d'image
            return None, None, []

        with torch.no_grad():
            logits = self.model(img_tensor)      # (1, 4)
            probs = torch.sigmoid(logits)[0]     # (4,)

        # Tensor -> numpy
        probs_np = probs.cpu().numpy()

        # Décision booléenne par seuil
        decisions = probs_np > self.threshold   # np.array bool (4,)

        # Calculer les changements par rapport à l'état précédent
        to_send = []
        for i, cmd in enumerate(self.commands):
            if decisions[i] != self.prev_decisions[i]:
                # True -> key down, False -> key up
                to_send.append((cmd, bool(decisions[i])))

        # Sauvegarder l'état courant pour la prochaine frame
        self.prev_decisions = decisions.copy()

        if self.debug:
            # Affichage au format de ton exemple
            probs_print = np.round(probs_np, 3)
            decisions_print = [bool(d) for d in decisions]
            print(f"probs={probs_print} decisions={decisions_print} to_send={to_send}")

        return probs_np, decisions, to_send

    def process_message(self, message, data_collector):
        """
        Callback pour DataCollectionUI :
        - message : sensing snapshot
        - data_collector : interface pour envoyer les commandes
        """
        _, _, to_send = self.nn_infer(message)

        # On envoie uniquement les changements
        for command, start in to_send:
            data_collector.onCarControlled(command, start)


if __name__ == "__main__":
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = NNMsgProcessor(threshold=0.3, debug=True)
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
