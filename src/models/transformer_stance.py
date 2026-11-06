import math

import torch
import torch.nn as nn


def sinusoidal_encoding(positions: torch.Tensor, d_model: int, max_len: int = 5000) -> torch.Tensor:
    """
    任意の（負を含む）整数位置インデックスに対する固定の正弦波 Positional Encoding を返す．

    transformer.py の PositionalEncoding と同一の式・同一の演算（同じ形状のテーブルを一度に計算）で
    作ったテーブルから行を引くため，非負の位置では既存の PositionalEncoding.pe の行とビット単位で一致する．
    負の位置 -p は sin 成分の符号反転（sin は奇関数，cos は偶関数）で得る．
    したがって同じ位置インデックスには，どの条件でも常に同一のベクトルが割り当てられる．

    Args:
        positions: (T,) 整数の位置インデックス（|p| < max_len）
        d_model:   埋め込み次元（偶数）
    Returns:
        (T, d_model)
    """
    pos = positions.long()
    if pos.abs().max() >= max_len:
        raise ValueError(f"|position| must be < {max_len}")
    full = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    table = torch.zeros(max_len, d_model)
    table[:, 0::2] = torch.sin(full * div_term)
    table[:, 1::2] = torch.cos(full * div_term)

    pe = table[pos.abs()].clone()
    sign = torch.where(pos < 0, -1.0, 1.0).unsqueeze(1)
    pe[:, 0::2] = pe[:, 0::2] * sign
    return pe


class IndexedPositionalEncoding(nn.Module):
    """
    任意の位置インデックス列を指定できる正弦波 Positional Encoding．
    位置インデックスはコンストラクタで固定する（学習パラメータなし）．
    """
    def __init__(self, d_model: int, positions: torch.Tensor):
        super().__init__()
        self.register_buffer('positions', positions.to(torch.float), persistent=False)
        self.register_buffer('pe', sinusoidal_encoding(positions, d_model).unsqueeze(0), persistent=False)

    def forward(self, x):
        # x: (Batch, SeqLen, d_model)
        if x.size(1) != self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} does not match positional indices {self.pe.size(1)}")
        return x + self.pe


class StanceExtractTransformer(nn.Module):
    """
    遊脚期を含む入力系列全体を Transformer Encoder に入力し，
    Encoder 出力のうち立脚期に対応する区間だけを取り出して床反力を推定するモデル．

        x: (B, T, input_dim)  -- T = 200 (HS-TO) / n_swing+200 (TO-TO, HS-HS)
        -> embedding * sqrt(d_model) + PE(位置インデックス)
        -> TransformerEncoder (全 T 点で Self-Attention)
        -> [:, stance_start : stance_start+stance_len]   (B, stance_len, d_model)
        -> FC (d_model -> 64 -> output_dim)
        -> (B, stance_len, output_dim)

    入力を 200 点へ圧縮する処理は行わない．
    層構成・初期化・スケーリングは transformer.py の TimeSeriesTransformer と同一
    （サブモジュール名も同一）．違いは PE の位置インデックスと Encoder 後の区間抽出のみ．

    Args:
        seq_len:       入力系列長 T
        stance_start:  立脚期の開始 index（TO-TO: n_swing, HS-HS / HS-TO: 0）
        stance_len:    立脚期の点数（200）
        pos_start:     系列先頭の位置インデックス．i 番目の点の位置は pos_start + i．
                       TO-TO: -n_swing（立脚期が 0..199），HS-TO/HS-HS: 0．
                       None の場合は -stance_start（= 立脚期を 0..199 に揃える既定設計）．
                       0 を指定すると従来方式（先頭から 0..T-1）になる．
    """
    def __init__(self, input_dim=14, output_dim=3, d_model=128, nhead=4, num_layers=3,
                 dim_feedforward=256, dropout_prob=0.1,
                 seq_len=200, stance_start=0, stance_len=200, pos_start=None, **kwargs):
        super().__init__()

        if stance_start < 0 or stance_start + stance_len > seq_len:
            raise ValueError(f"stance window [{stance_start}, {stance_start + stance_len}) "
                             f"is outside the sequence length {seq_len}")
        if pos_start is None:
            pos_start = -stance_start

        self.input_dim = input_dim
        self.d_model = d_model
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.stance_start = stance_start
        self.stance_len = stance_len
        self.pos_start = pos_start

        self.embedding = nn.Linear(self.input_dim, self.d_model)
        positions = torch.arange(seq_len) + pos_start
        self.pos_encoder = IndexedPositionalEncoding(self.d_model, positions)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

    def encode(self, x):
        """Encoder 全出力 (B, T, d_model)．形状確認・解析用．"""
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        return self.transformer_encoder(x)

    def forward(self, x):
        r_out = self.encode(x)                                                  # (B, T, d_model)
        r_stance = r_out[:, self.stance_start:self.stance_start + self.stance_len, :]  # (B, 200, d_model)
        return self.fc(r_stance)                                                # (B, 200, output_dim)
