import os, sys
import torch
import numpy as np
import collections
from PyQt6 import QtWidgets

# ---------------------------------------------------------
# Add project root so imports work
# ---------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from scripts.model_c1.cnn_lstm import CNNLSTM
from data_collector import DataCollectionUI

from PIL import Image

MODEL_PATH = "scripts/train/all_best_cnn_lstm.pth"
THRESHOLD = 0.18

# ---------------------------------------------------------
# Autopilot CNN + LSTM
# ---------------------------------------------------------
class CNNLSTMAutopilot:
    def __init__(self, model_path=MODEL_PATH,
                 seq_len=10, img_w=160, img_h=120):

        self.seq_len = seq_len
        self.img_w = img_w
        self.img_h = img_h

        # Keep last 10 frames
        self.frame_buffer = collections.deque(maxlen=seq_len)

        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Autopilot device:", self.device)

        # Load model
        self.model = CNNLSTM(num_actions=4).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    # -----------------------------------------------------
    # PREPROCESS IMAGE — EXACT SAME AS TRAINING
    # -----------------------------------------------------
    def preprocess_image(self, img):
        """
        img: numpy array (H, W, 3) already in RGB from the simulator.
        """

        # Convert numpy → PIL
        img = Image.fromarray(img)

        # Resize like in training
        img = img.resize((self.img_w, self.img_h), Image.BILINEAR)

        # Back to numpy, normalized to [0,1]
        img = np.array(img, dtype=np.float32) / 255.0

        return img  # shape: (H,W,3)

    # -----------------------------------------------------
    # MODEL INFERENCE
    # -----------------------------------------------------
    def nn_infer(self):
        if len(self.frame_buffer) < self.seq_len:
            return None  # not enough frames yet

        # (seq, H, W, 3) → (1, seq, H, W, 3)
        frames = np.array(self.frame_buffer, dtype=np.float32)
        frames = frames[np.newaxis, ...]

        x = torch.from_numpy(frames).to(self.device)

        with torch.no_grad():
            outputs = self.model(x)[0].cpu().numpy()

        # Debug prints so we know what the model outputs
        print("raw outputs:", outputs)

        w, s, a, d = outputs

        # Use softer threshold (model outputs are often around 0.2–0.8)
        return {
            "forward": w > THRESHOLD,
            "back":    s > THRESHOLD,
            "left":    a > THRESHOLD,
            "right":   d > THRESHOLD,
        }

    # -----------------------------------------------------
    # CALLBACK FOR SIMULATOR — CALLED FOR EACH FRAME
    # -----------------------------------------------------
    def process_message(self, message, data_collector):
        img = message.image  # raw numpy RGB image

        # Preprocess
        frame = self.preprocess_image(img)
        self.frame_buffer.append(frame)

        # Run inference
        controls = self.nn_infer()
        if controls is None:
            return

        # Send commands to simulator
        for key, pressed in controls.items():
            data_collector.onCarControlled(key, pressed)


# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
if __name__ == "__main__":

    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    # Load autopilot model
    autopilot = CNNLSTMAutopilot(
        model_path=MODEL_PATH,
        seq_len=10
    )

    # Create DataCollector window
    data_window = DataCollectionUI(autopilot.process_message)
    data_window.show()

    app.exec()
