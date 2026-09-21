"""
從 src/SoftPINCH/src/models/classification_pipeline.py 逐字搬過來的模型架構定義。

不能直接 import 該檔案，因為它頂端 `import sherpa` 跟
`from src.utilities.load_and_visualize_data import load_datasets` 這兩個依賴
在這個 repo 裡不存在/沒裝（sherpa 是他們拿來做 hyperparameter tuning 的套件，
load_and_visualize_data.py 這個檔案本身沒有隨 repo 公開）。這裡只複製跑
inference 真正需要的類別（CNN / LSTM / DenseLayer / SingleNet_CNN_LSTM），
架構要跟 checkpoint 裡存的 state_dict 完全對得上才能載入權重。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_dim: int, cnn_filters: int, kernel_size: int,
                 activation: type[nn.Module], dropout: float):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=cnn_filters, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_features=cnn_filters),
            activation(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=cnn_filters, out_channels=cnn_filters * 2, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(num_features=cnn_filters * 2),
            activation(),
            nn.MaxPool1d(kernel_size=2),

            nn.Dropout(dropout),
        )

    def forward(self, data: torch.Tensor):
        x = data.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        return x


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers,
                             batch_first=True, dropout=dropout if layers > 1 else 0,
                             bidirectional=bidirectional)

    def forward(self, data: torch.Tensor):
        lstm_out, (hn, _) = self.lstm(data)
        hn = self._extract_hidden(hn)
        return lstm_out, hn

    def _extract_hidden(self, units: torch.Tensor):
        if self.bidirectional:
            h_forward = units[-2]
            h_backward = units[-1]
            h_final = torch.cat((h_forward, h_backward), dim=1)
        else:
            h_final = units[-1]
        return h_final


class DenseLayer(nn.Module):
    def __init__(self, lstm_hidden_dim: int, output_dim: int, dense_ratio: float,
                 activation: type[nn.Module], dropout: float):
        super().__init__()
        dense_layers = max(8, int(lstm_hidden_dim * dense_ratio))
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden_dim, dense_layers),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(dense_layers, output_dim),
        )

    def forward(self, lstm_units: torch.Tensor):
        return self.classifier(lstm_units)


class SingleNet_CNN_LSTM(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, lstm_layers: int,
                 bidirectional: bool, dropout: float, activation: str, dense_ratio: float,
                 cnn_filters: int, kernel_size: int):
        super().__init__()
        lstm_hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim
        activations = {'relu': nn.ReLU, 'elu': nn.ELU}
        act = activations[activation]

        self.cnn = CNN(input_dim=input_dim, cnn_filters=cnn_filters, kernel_size=kernel_size,
                        activation=act, dropout=dropout)
        self.lstm = LSTM(input_dim=cnn_filters * 2, hidden_dim=hidden_dim, layers=lstm_layers,
                          dropout=dropout, bidirectional=bidirectional)
        self.classifier = DenseLayer(lstm_hidden_dim=lstm_hidden_dim, output_dim=output_dim,
                                      dense_ratio=dense_ratio, activation=act, dropout=dropout)

    def forward(self, data: torch.Tensor):
        cnn_out = self.cnn(data)
        _, hn = self.lstm(cnn_out)
        logits = self.classifier(hn)
        return logits, None, None


def load_model(model_pth_path):
    checkpoint = torch.load(model_pth_path, map_location="cpu", weights_only=False)
    model = SingleNet_CNN_LSTM(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
