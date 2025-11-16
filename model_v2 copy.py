# model.py
import torch
import torch.nn as nn

from config import NUM_OUTPUTS, NUM_CHANNELS


class SimpleCNN(nn.Module):
    """
    CNN très simple pour prédire 4 commandes (F, B, L, R)
    à partir d'une image RGB prétraitée.
    """
    def __init__(self, num_outputs=NUM_OUTPUTS):
        super().__init__()

        self.features = nn.Sequential(
            # Conv 1 : 3 -> 32
            nn.Conv2d(NUM_CHANNELS, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            # Conv 2 : 32 -> 64
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            # Conv 3 : 64 -> 96
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Sortie fixe en 4x4
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Taille après flatten = 96 * 4 * 4 = 1536
        flatten_size = 96 * 4 * 4

        # Partie fully-connected
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # (B, 96, 4, 4) -> (B, 1536)

            nn.Linear(flatten_size, 128), # 1536 -> 128
            nn.ReLU(inplace=True),

            nn.Linear(128, 64),           # 128 -> 64
            nn.ReLU(inplace=True),

            nn.Linear(64, num_outputs),   # 64 -> 4 (logits)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_outputs=NUM_OUTPUTS):
    return SimpleCNN(num_outputs=num_outputs)
