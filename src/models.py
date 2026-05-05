"""
models.py — 模型定义
=====================
- LSTMModel1:  单层 LSTM (hidden=64) -> FC
- LSTMModel2:  双层 LSTM + Dropout -> FC
- TransformerModel: PositionalEncoding + TransformerEncoder -> mean pool -> FC
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMModel1(nn.Module):
    """单层 LSTM + 全连接输出"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel2(nn.Module):
    """双层 LSTM + Dropout + 全连接输出"""

    def __init__(self, input_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class PositionalEncoding(nn.Module):
    """正弦-余弦位置编码 (不可学习)"""

    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Transformer 编码器: 输入投影 -> 位置编码 -> Encoder -> 均值池化 -> FC"""

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)


def build_model(cfg):
    """从 config 构建模型"""
    name = cfg["model"]["name"]
    d = cfg["model"]
    if name == "lstm1":
        return LSTMModel1(d["input_dim"], d["hidden_dim"])
    elif name == "lstm2":
        return LSTMModel2(d["input_dim"], d["hidden_dim"], d["dropout"])
    elif name == "transformer":
        return TransformerModel(d["input_dim"], d["d_model"], d["nhead"],
                                d["num_layers"], d["dim_feedforward"], d["dropout"])
    raise ValueError(f"Unknown model: {name}")


def build_loss(name):
    if name == "mae":
        return nn.L1Loss()
    elif name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {name}")
