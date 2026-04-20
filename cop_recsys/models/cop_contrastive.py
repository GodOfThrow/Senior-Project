from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import CLSTransformerEncoder
from ..time_decay import build_consecutive_pairs


class COPContrastive(nn.Module):
    def __init__(
        self,
        item_size: int,
        d_model: int = 128,
        pad_id: int = 0,
        temperature: float = 0.07,
        detach_bank: bool = True,
        use_time_decay: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d = d_model
        self.tau = temperature
        self.detach_bank = detach_bank
        self.use_time_decay = use_time_decay
        self.num_items = item_size

        self.enc = CLSTransformerEncoder(item_size=item_size, d_model=d_model, pad_id=pad_id)

    @staticmethod
    def _avg_cosine_topk(
        anchor_pairs: torch.Tensor,
        anchor_pair_mask: torch.BoolTensor,
        bank_pairs: torch.Tensor,
        bank_pair_mask: torch.BoolTensor,
        topk: int,
        anchor_pair_weights: torch.Tensor | None = None,
        bank_pair_weights: torch.Tensor | None = None,
        eps: float = 1e-8,
    ):
        del bank_pair_weights
        anchor_norm = F.normalize(anchor_pairs, dim=-1, eps=eps)
        bank_norm = F.normalize(bank_pairs, dim=-1, eps=eps)

        sim_pos = (anchor_norm.unsqueeze(1) * bank_norm.unsqueeze(0)).sum(-1)
        common = (anchor_pair_mask.unsqueeze(1) & bank_pair_mask.unsqueeze(0)).float()

        if anchor_pair_weights is None:
            anchor_pair_weights = anchor_pair_mask.float()

        wa = anchor_pair_weights.unsqueeze(1).clamp_min(0.0)
        w_pos = wa * common

        num = (sim_pos * w_pos).sum(-1)
        den = anchor_pair_mask.float().sum(-1, keepdim=True).clamp_min(1.0)
        sims = num / den

        k = min(topk, sims.size(1))
        vals, idx = torch.topk(sims, k=k, dim=1, largest=True)
        return idx, vals, sims

    @staticmethod
    def _avg_l2_topk(
        anchor_pairs: torch.Tensor,
        anchor_pair_mask: torch.BoolTensor,
        bank_pairs: torch.Tensor,
        bank_pair_mask: torch.BoolTensor,
        topk: int,
        eps: float = 1e-8,
    ):
        diff = anchor_pairs.unsqueeze(1) - bank_pairs.unsqueeze(0)
        dist_pos = torch.sqrt((diff * diff).sum(-1) + eps)

        common = (anchor_pair_mask.unsqueeze(1) & bank_pair_mask.unsqueeze(0)).float()
        num = (dist_pos * common).sum(-1)
        den = common.sum(-1).clamp_min(1.0)
        dists = num / den

        k = min(topk, dists.size(1))
        vals, idx = torch.topk(dists, k=k, dim=1, largest=False)
        return idx, vals, dists

    def _mp_info_nce(
        self,
        cls_anchor: torch.Tensor,
        cls_neighbors: torch.Tensor,
        cls_negatives: torch.Tensor,
        pos_weights: torch.Tensor | None = None,
    ):
        if cls_neighbors.numel() == 0:
            return cls_anchor.new_tensor(0.0, requires_grad=True), {"logits_pos": None, "logits_neg": None}

        a = F.normalize(cls_anchor, dim=-1)
        p = F.normalize(cls_neighbors, dim=-1)
        n = F.normalize(cls_negatives, dim=-1)

        logits_pos = (a.unsqueeze(1) * p).sum(-1) / self.tau
        logits_neg = (a.unsqueeze(1) * n).sum(-1) / self.tau

        logsumexp_pos = torch.logsumexp(logits_pos, dim=1)
        logsumexp_neg = torch.logsumexp(logits_neg, dim=1)
        log_denom = torch.logaddexp(logsumexp_pos, logsumexp_neg)

        if pos_weights is None:
            log_num = torch.logsumexp(logits_pos, dim=1)
        else:
            w = pos_weights.clamp_min(1e-12)
            w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
            log_num = torch.logsumexp(torch.log(w + 1e-12) + logits_pos, dim=1)

        loss = -(log_num - log_denom).mean()
        return loss, {"logits_pos": logits_pos.detach(), "logits_neg": logits_neg.detach()}

    def encode_only(self, items: torch.LongTensor, timestamps: torch.Tensor):
        del timestamps
        attn_mask = items.ne(self.pad_id)
        cls, _ = self.enc(items, attn_mask)
        return {"cls": cls}

    def forward(
        self,
        items: torch.LongTensor,
        timestamps: torch.Tensor,
        *,
        bank_is_batch: bool = False,
        bank_items: torch.LongTensor | None = None,
        bank_timestamps: torch.Tensor | None = None,
        topk: int = 5,
        num_negatives: int = 5,
        floor: float = 1e-3,
        predict_mode: str = "linear",
        metric: str = "cosine",
        return_loss: bool = True,
    ):
        attn_mask = items.ne(self.pad_id)
        cls_anchor, E_anchor = self.enc(items, attn_mask)

        t = timestamps.to(torch.float32)
        gaps = torch.zeros_like(t)
        gaps[:, 1:] = (t[:, 1:] - t[:, :-1]).clamp_min(0)
        pairs_anchor, w_pairs_a, mask_pairs_a = build_consecutive_pairs(
            E_anchor,
            attn_mask,
            gaps,
            use_time_decay=self.use_time_decay,
            floor=floor,
            predict_mode=predict_mode,
        )

        if bank_is_batch:
            cls_bank, pairs_bank, w_pairs_b, mask_pairs_b = cls_anchor, pairs_anchor, w_pairs_a, mask_pairs_a
        else:
            assert bank_items is not None and bank_timestamps is not None, (
                "bank_items and bank_timestamps are required when bank_is_batch=False"
            )
            bank_mask = bank_items.ne(self.pad_id)
            cls_bank, E_bank = self.enc(bank_items, bank_mask)

            tb = bank_timestamps.to(torch.float32)
            gaps_b = torch.zeros_like(tb)
            gaps_b[:, 1:] = (tb[:, 1:] - tb[:, :-1]).clamp_min(0)
            pairs_bank, w_pairs_b, mask_pairs_b = build_consecutive_pairs(
                E_bank,
                bank_mask,
                gaps_b,
                use_time_decay=self.use_time_decay,
                floor=floor,
                predict_mode=predict_mode,
            )

        La = pairs_anchor.size(1)
        Lb = pairs_bank.size(1)
        Lp = min(La, Lb)
        if Lp == 0:
            out = {
                "cls_anchor": cls_anchor,
                "pairs_anchor": pairs_anchor,
                "pair_weights": w_pairs_a,
                "pair_weights_aligned": None,
                "topk_idx": None,
                "topk_score": None,
                "topk_dist": None,
                "topk_sim": None,
                "all_scores": None,
                "cls_neighbors": cls_anchor.new_zeros(cls_anchor.size(0), 0, cls_anchor.size(1)),
                "cls_negatives": cls_anchor.new_zeros(cls_anchor.size(0), 0, cls_anchor.size(1)),
            }
            if not return_loss:
                return out
            loss = cls_anchor.sum() * 0.0
            out.update({"loss": loss, "logs": {"logits_pos": None, "logits_neg": None}})
            return out

        A, Am = pairs_anchor[:, -Lp:, :], mask_pairs_a[:, -Lp:]
        Bp, Bm = pairs_bank[:, -Lp:, :], mask_pairs_b[:, -Lp:]
        Aw = w_pairs_a[:, -Lp:]
        Bw = w_pairs_b[:, -Lp:]

        metric = metric.lower()
        topk_idx = topk_score = topk_dist = topk_sim = all_scores = None
        cls_neighbors = cls_anchor.new_zeros(cls_anchor.size(0), 0, cls_anchor.size(1))
        cls_negatives = cls_anchor.new_zeros(cls_anchor.size(0), 0, cls_anchor.size(1))
        pos_weights = None

        if topk > 0:
            if metric == "l2":
                A_l2 = A * Aw.unsqueeze(-1)
                B_l2 = Bp * Bw.unsqueeze(-1)
                topk_idx, topk_dist, all_dists = self._avg_l2_topk(A_l2, Am, B_l2, Bm, topk=topk)
                all_scores = -all_dists
                topk_score = torch.exp(-topk_dist)
                topk_sim = topk_score
                cls_neighbors = cls_bank[topk_idx]
                pos_weights = topk_score

                k_neg = min(num_negatives, all_dists.size(1))
                _, neg_idx = torch.topk(all_dists, k=k_neg, dim=1, largest=True)
                cls_negatives = cls_bank[neg_idx]

            elif metric in ("cos", "cosine", "cosine_similarity"):
                topk_idx, topk_sim_raw, all_sims = self._avg_cosine_topk(
                    A,
                    Am,
                    Bp,
                    Bm,
                    topk=topk,
                    anchor_pair_weights=Aw,
                    bank_pair_weights=Bw,
                )
                all_scores = all_sims
                topk_score = topk_sim_raw
                cls_neighbors = cls_bank[topk_idx]
                pos_weights = (topk_sim_raw + 1.0) / 2.0
                topk_sim = pos_weights

                k_neg = min(num_negatives, all_sims.size(1))
                _, neg_idx = torch.topk(all_sims, k=k_neg, dim=1, largest=False)
                cls_negatives = cls_bank[neg_idx]
            else:
                raise ValueError(f"Unsupported metric: {metric}. Choose from ['l2', 'cosine'].")

            if self.detach_bank:
                cls_neighbors = cls_neighbors.detach()
                cls_negatives = cls_negatives.detach()
                topk_score = topk_score.detach()
                if topk_dist is not None:
                    topk_dist = topk_dist.detach()
                if topk_sim is not None:
                    topk_sim = topk_sim.detach()
                if all_scores is not None:
                    all_scores = all_scores.detach()

        out = {
            "cls_anchor": cls_anchor,
            "pairs_anchor": pairs_anchor,
            "pair_weights": w_pairs_a,
            "pair_weights_aligned": Aw,
            "topk_idx": topk_idx,
            "topk_score": topk_score,
            "topk_dist": topk_dist,
            "topk_sim": topk_sim,
            "all_scores": all_scores,
            "cls_neighbors": cls_neighbors,
            "cls_negatives": cls_negatives,
            "use_time_decay": self.use_time_decay,
        }

        if not return_loss:
            return out

        loss_ctr, logs = self._mp_info_nce(
            cls_anchor=cls_anchor,
            cls_neighbors=cls_neighbors,
            cls_negatives=cls_negatives,
            pos_weights=pos_weights if topk > 0 else None,
        )
        out["loss"] = loss_ctr
        out["logs"] = logs
        return out
