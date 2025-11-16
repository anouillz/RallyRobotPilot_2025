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
            nn.Conv2d(NUM_CHANNELS, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4)),  # -> (batch, 32, 4, 4)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),               
            nn.Linear(32*4*4, 32),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_outputs), # logits (pas de sigmoid ici)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_outputs=NUM_OUTPUTS):
    return SimpleCNN(num_outputs=num_outputs)
