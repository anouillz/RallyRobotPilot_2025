import torch
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        h = self.conv(x)
        return self.fc(h)


class CNNLSTMPolicy(nn.Module):
    def __init__(self, feature_dim=256, lstm_hidden=256, use_lstm=False):
        super().__init__()
        self.encoder = CNNEncoder(in_channels=3, feature_dim=feature_dim)
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(feature_dim, lstm_hidden, batch_first=True)
            head_in = lstm_hidden
        else:
            head_in = feature_dim

        self.head = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, H, W) if no LSTM
        # or x: (B, T, C, H, W) if use_lstm
        if self.use_lstm:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            feats = self.encoder(x)
            feats = feats.view(B, T, -1)
            outs, _ = self.lstm(feats)
            # take last output
            last = outs[:, -1, :]
            return self.head(last)
        else:
            feats = self.encoder(x)
            return self.head(feats)


if __name__ == '__main__':
    import torch
    m = CNNLSTMPolicy(use_lstm=False)
    x = torch.randn(2, 3, 120, 160)
    y = m(x)
    print(y.shape)
