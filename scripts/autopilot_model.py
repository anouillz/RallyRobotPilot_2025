#!/usr/bin/env python3
"""Autopilot that runs a trained PyTorch model and sends key commands to the simulator.

Usage:
  python3 scripts/autopilot_model.py --checkpoint scripts/training/checkpoints/bc_epoch1.pt

The script hooks into the existing `DataCollectionUI` message loop (same as
`example_autopilot.py`) and calls `data_collector.onCarControlled(direction, start)`
to push/release controls.

It loads the model class from `scripts/training/models.py` using a file loader so
it doesn't rely on package imports.
"""
import argparse
import importlib.util
import sys
import os
from types import ModuleType

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import time

from data_collector import DataCollectionUI


def load_module_from_path(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("model_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ModelAutopilot:
    def __init__(self, checkpoint_path: str, model_file: str, device: str = None, img_size=(160, 120), threshold=0.5):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size  # (width, height)
        self.threshold = float(threshold)

        # load model code
        mod = load_module_from_path(model_file)
        # expect class CNNLSTMPolicy in the module
        if not hasattr(mod, 'CNNLSTMPolicy'):
            raise RuntimeError(f"Model file {model_file} does not define CNNLSTMPolicy")

        ModelClass = getattr(mod, 'CNNLSTMPolicy')
        self.model = ModelClass(use_lstm=False)
        # load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state' in ckpt:
            state = ckpt['model_state']
        else:
            state = ckpt
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # transform
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size[1], self.img_size[0])),
            transforms.ToTensor(),
        ])
        
        # recorded control order: (W, S, A, D)
        self.prev = [0, 0, 0, 0]
        self.dir_map = {0: 'forward', 1: 'back', 2: 'left', 3: 'right'}

    def nn_infer(self, message):
        img = getattr(message, 'image', None)
        if img is None:
            return []

        if isinstance(img, np.ndarray):
            pil = Image.fromarray(img)
        else:
            pil = Image.fromarray(np.asarray(img))

        x = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            preds = self.model(x)
            # preds are sigmoids
            probs = preds.squeeze(0).cpu().numpy()

        # binary decisions
        decisions = (probs >= self.threshold).astype(int).tolist()

        # Debug print: probabilities and decisions
        try:
            print(f"[Autopilot] probs={probs.tolist()} threshold={self.threshold} -> decisions={decisions}")
        except Exception:
            pass

        commands = []
        # emit push/release based on change from prev
        for i, val in enumerate(decisions):
            if val != self.prev[i]:
                # if new val is 1 -> push, else release
                cmd = (self.dir_map[i], bool(val))
                commands.append(cmd)
        self.prev = decisions
        return commands

    def process_message(self, message, data_collector: DataCollectionUI):
        commands = self.nn_infer(message)
        for direction, start in commands:
            # Print every command that will be sent to the main UI/simulator
            now = time.time()
            action = 'push' if start else 'release'
            print(f"[Autopilot] {now:.3f} -> sending: {action} {direction}")
            data_collector.onCarControlled(direction, start)


def main():
    p = argparse.ArgumentParser(description='Run model-based autopilot')
    p.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pt)')
    p.add_argument('--model-file', default='scripts/training/models.py', help='Path to model class file')
    p.add_argument('--width', type=int, default=160, help='Input image width')
    p.add_argument('--height', type=int, default=120, help='Input image height')
    p.add_argument('--threshold', type=float, default=0.5, help='Sigmoid threshold')
    p.add_argument('--device', default=None, help='torch device string (e.g. cpu or cuda:0)')
    args = p.parse_args()

    # instantiate UI and autopilot
    autopilot = ModelAutopilot(args.checkpoint, args.model_file, device=args.device, img_size=(args.width, args.height), threshold=args.threshold)

    import sys
    from PyQt6 import QtWidgets

    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)
    data_window = DataCollectionUI(autopilot.process_message)
    data_window.show()
    app.exec()


if __name__ == '__main__':
    main()
