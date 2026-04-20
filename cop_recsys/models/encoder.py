from __future__ import annotations

import torch
import torch.nn as nn


class CLSTransformerEncoder(nn.Module):
    """
    items[B,L] + attn_mask[B,L] -> cls[B,D], tokens[B,L,D]
    """

    def __init__(
        self,
        item_size: int,
        d_model: int = 128,
        pad_id: int = 0,
        max_len: int = 512,
        n_layers: int = 2,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        norm_first: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_len = max_len
        self.causal = causal
        self.num_items = item_size

        self.item_emb = nn.Embedding(item_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.cls_tok = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.xavier_uniform_(self.pos_emb.weight)
        with torch.no_grad():
            self.item_emb.weight[self.pad_id].zero_()

    @staticmethod
    def _causal_mask(L: int, device: torch.device) -> torch.BoolTensor:
        return torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), 1)

    def forward(self, items: torch.LongTensor, attn_mask: torch.BoolTensor):
        B, L = items.shape
        device = items.device

        L_clip = min(L, self.max_len)
        pos = torch.arange(L_clip, device=device).unsqueeze(0).expand(B, L_clip)

        tok = self.item_emb(items[:, :L_clip]) + self.pos_emb(pos)
        cls = self.cls_tok.expand(B, 1, -1)
        X = torch.cat([cls, tok], dim=1)

        key_padding_tok = ~attn_mask[:, :L_clip]
        key_padding = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), key_padding_tok],
            dim=1,
        )
        src_mask = self._causal_mask(X.size(1), device) if self.causal else None

        H = self.encoder(
            self.drop(self.norm(X)),
            mask=src_mask,
            src_key_padding_mask=key_padding,
        )
        cls_out = H[:, 0, :]
        tok_out = H[:, 1:, :]
        return cls_out, tok_out
