import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNLSTM(nn.Module):
    def __init__(self, num_actions=4, hidden_size=128):
        super().__init__()

        # cnn for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2),  # (32, 58, 78)
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=5, stride=2), # (64, 27, 37)
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2), # (64, 13, 18)
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2), # (128, 6, 8)
            nn.ReLU()
        )

        # compute CNN output size dynamically
        dummy = torch.zeros(1, 3, 120, 160)
        out = self.cnn(dummy)
        cnn_feature_size = out.view(1, -1).shape[1]

        # lstm for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_feature_size,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.3,
            batch_first=True
        )

        # final fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_actions),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (B, seq_len, H, W, C)
        """

        B, seq_len, H, W, C = x.shape

        x = x.permute(0, 1, 4, 2, 3)  # (B, seq, C, H, W)

        cnn_features = []
        for t in range(seq_len):
            f = self.cnn(x[:, t])
            f = f.reshape(B, -1)
            cnn_features.append(f)

        cnn_features = torch.stack(cnn_features, dim=1)

        lstm_out, _ = self.lstm(cnn_features)

        last_output = lstm_out[:, -1]

        return self.fc(last_output)
