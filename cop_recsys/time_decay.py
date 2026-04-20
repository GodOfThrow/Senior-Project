from __future__ import annotations

import numpy as np
import torch


def isotonic_decreasing_fit(
    x,
    y,
    sample_weight=None,
    return_sort_idx: bool = True,
):
    """
    Fit a non-increasing isotonic regression y_hat(x).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if sample_weight is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)

    sort_idx = np.argsort(x, kind="mergesort")
    xs = x[sort_idx]
    ys = y[sort_idx]
    ws = w[sort_idx]

    means = []
    weights = []
    starts = []

    for i in range(n):
        means.append(ys[i])
        weights.append(ws[i])
        starts.append(i)

        while len(means) >= 2 and means[-2] < means[-1]:
            w1, w2 = weights[-2], weights[-1]
            m1, m2 = means[-2], means[-1]
            new_w = w1 + w2
            new_m = (w1 * m1 + w2 * m2) / new_w
            weights[-2] = new_w
            means[-2] = new_m
            starts.pop()
            weights.pop()
            means.pop()

    y_hat_sorted = np.empty(n, dtype=float)
    for b in range(len(means)):
        i0 = starts[b]
        i1 = starts[b + 1] if b + 1 < len(starts) else n
        y_hat_sorted[i0:i1] = means[b]

    if return_sort_idx:
        return xs, y_hat_sorted, sort_idx
    return xs, y_hat_sorted


def isotonic_predict(x_query, x_fit, y_hat, mode: str = "linear"):
    x_query = np.asarray(x_query, dtype=float)
    x_fit = np.asarray(x_fit, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)

    if mode == "linear":
        xu, idx = np.unique(x_fit, return_index=True)
        yu = y_hat[idx]
        return np.interp(x_query, xu, yu, left=yu[0], right=yu[-1])

    if mode == "step":
        xu, idx = np.unique(x_fit, return_index=True)
        yu = y_hat[idx]
        pos = np.searchsorted(xu, x_query, side="right") - 1
        pos = np.clip(pos, 0, len(xu) - 1)
        return yu[pos]

    raise ValueError("mode must be 'linear' or 'step'")


def rank_targets(x: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    """
    Smallest gap -> 1.0, largest gap -> floor.
    """
    n = len(x)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([1.0], dtype=float)

    order = np.argsort(x)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    y = 1.0 - (ranks - 1) * (1.0 - floor) / (n - 1)
    return y


def build_consecutive_pairs(
    E: torch.Tensor,
    attn_mask: torch.BoolTensor,
    gaps_sec: torch.Tensor | None = None,
    *,
    use_time_decay: bool = True,
    floor: float = 1e-3,
    predict_mode: str = "linear",
):
    """
    Returns:
      pairs     : [B,L-1,D] raw consecutive pair features
      weights   : [B,L-1] pair weights
      pair_mask : [B,L-1] valid-pair mask
    """
    B, L, D = E.shape
    device, dtype = E.device, E.dtype

    pair_mask = attn_mask[:, 1:] & attn_mask[:, :-1]
    pairs = (E[:, 1:, :] * E[:, :-1, :]) * pair_mask.unsqueeze(-1)

    if not use_time_decay:
        return pairs, pair_mask.to(dtype=dtype), pair_mask

    if gaps_sec is None:
        raise ValueError("gaps_sec is required when use_time_decay=True")

    pair_gaps = gaps_sec[:, 1:]
    w_all = torch.zeros(B, L - 1, device=device, dtype=dtype)

    for b in range(B):
        valid = pair_mask[b].nonzero(as_tuple=False).squeeze(-1)
        if valid.numel() == 0:
            continue
        x = pair_gaps[b, valid].detach().cpu().numpy().astype(float)
        y = rank_targets(x, floor=floor)
        x_fit, y_hat, _ = isotonic_decreasing_fit(x, y)
        w = isotonic_predict(x, x_fit, y_hat, mode=predict_mode)
        w_all[b, valid] = torch.from_numpy(w).to(device=device, dtype=dtype)

    w_all = w_all * pair_mask.to(dtype=dtype)
    return pairs, w_all, pair_mask
