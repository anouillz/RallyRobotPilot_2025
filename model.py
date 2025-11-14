# model.py
import torch
import torch.nn as nn

from config import NUM_OUTPUTS, NUM_CHANNELS


class SimpleCNN(nn.Module):
    """
    CNN très simple pour prédire 4 commandes (F, B, L, R)
    à partir d'une image RGB.
    """
    def __init__(self, num_outputs=NUM_OUTPUTS):
        super().__init__()

        # Bloc convolution très léger
        self.features = nn.Sequential(
            # 1ère conv
            nn.Conv2d(NUM_CHANNELS, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            # 2ème conv
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            # Pooling adaptatif pour ne pas dépendre de la taille exacte de l'image
            nn.AdaptiveAvgPool2d((4, 4)),   # => (batch, 32, 4, 4)
        )

        # 32 * 4 * 4 = 512
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_outputs),  # BCEWithLogitsLoss
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_outputs=NUM_OUTPUTS):
    return SimpleCNN(num_outputs=num_outputs)
