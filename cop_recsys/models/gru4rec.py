from __future__ import annotations

import torch
import torch.nn as nn


class _GRUEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.item_emb = nn.Embedding(num_items, emb_dim, padding_idx=pad_id)
        self.gru = nn.GRU(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.pad_id = pad_id
        nn.init.xavier_uniform_(self.item_emb.weight)

    def forward(self, items: torch.LongTensor) -> torch.Tensor:
        x = self.item_emb(items)
        mask = items.ne(self.pad_id)
        lengths = mask.sum(dim=1).clamp_min(1)
        out, _ = self.gru(x)
        idx = (lengths - 1).view(-1, 1, 1).expand(out.size(0), 1, out.size(-1))
        return out.gather(1, idx).squeeze(1)


class GRU4RecBaseline(nn.Module):
    def __init__(
        self,
        num_items: int,
        emb_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
        pad_id: int = 0,
        tie_out: bool = True,
    ):
        super().__init__()
        self.enc = _GRUEncoder(num_items, emb_dim, hidden_dim, num_layers, dropout, pad_id)
        self.proj = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.item_bias = nn.Parameter(torch.zeros(num_items))
        self.num_items = num_items
        if tie_out and hidden_dim == emb_dim:
            with torch.no_grad():
                self.proj.weight.copy_(torch.eye(emb_dim))

    def encode_only(self, items: torch.LongTensor) -> torch.Tensor:
        return self.proj(self.enc(items))

    def full_scores(self, items: torch.LongTensor) -> torch.Tensor:
        cls = self.encode_only(items)
        return cls @ self.enc.item_emb.weight.t() + self.item_bias
