from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch


def load_movielens_ratings(path: str = "ratings.dat") -> pd.DataFrame:
    cols = ["UserID", "MovieID", "Rating", "Timestamp"]
    df = pd.read_csv(path, sep="::", engine="python", names=cols)
    df["Datetime"] = pd.to_datetime(df["Timestamp"], unit="s")
    df = df.sort_values(["UserID", "Datetime"]).reset_index(drop=True)
    df["TimeGap"] = df.groupby("UserID")["Datetime"].diff().dt.total_seconds()
    return df


def find_latest_session_from_tail(
    user_seq: pd.DataFrame,
    L: int = 10,
    k: float = 3.0,
    min_abs: int = 60,
    ratio: float = 1.25,
    delta: float = 0.0,
    relax: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = user_seq.sort_values("Datetime").reset_index(drop=True).copy()
    df["pos"] = np.arange(len(df))
    g = df["TimeGap"].fillna(0).to_numpy()
    N = len(df)
    pos_vals = g[g > 0]
    med = np.median(pos_vals) if len(pos_vals) else 0.0
    mad = (np.median(np.abs(pos_vals - med)) * 1.4826) if len(pos_vals) else 0.0
    thr = max(min_abs, med + k * max(mad, 1e-9))
    sufmax = np.full(N + 1, -np.inf)
    for i in range(N - 1, -1, -1):
        sufmax[i] = max(sufmax[i + 1], g[i])

    def try_with_ratio(r: float):
        for c in range(N - L - 1, -1, -1):
            inside_max = sufmax[c + 1] if c + 1 < N else -np.inf
            cond_thresh = g[c] >= thr
            cond_dom = g[c] >= inside_max * r + delta
            if cond_thresh and cond_dom:
                return c, r, inside_max
        return None

    ans = try_with_ratio(ratio)
    r_used = ratio
    if ans is None and relax:
        r = ratio
        while r > 1.0 and ans is None:
            r = max(1.0, r - 0.05)
            ans = try_with_ratio(r)
            r_used = r

    if ans is None:
        eligible = np.arange(0, max(N - L, 0))
        elig = eligible[g[eligible] >= thr]
        if len(elig):
            c = int(elig[np.argmax(g[elig])])
            inside_max = sufmax[c + 1] if c + 1 < N else -np.inf
        else:
            c = N - L - 1
            inside_max = sufmax[c + 1] if 0 <= c + 1 < N else -np.inf
    else:
        c, r_used, inside_max = ans

    df["is_break"] = False
    if c >= 0:
        df.loc[c, "is_break"] = True
    df["InLatestSession"] = df["pos"] > c
    df["LatestSessionId"] = np.where(df["InLatestSession"], 0, -1)

    meta = {
        "cut_pos": int(c),
        "cut_gap": float(g[c]) if 0 <= c < N else None,
        "inside_max": float(inside_max) if np.isfinite(inside_max) else None,
        "threshold": float(thr),
        "ratio_used": float(r_used),
    }
    return df, meta


def extract_latest_sessions(
    all_df: pd.DataFrame,
    L: int = 10,
    k: float = 3.0,
    min_abs: int = 60,
    ratio: float = 1.25,
    relax: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    latest_parts: list[pd.DataFrame] = []
    metas: list[dict[str, Any]] = []
    for uid, g in all_df.groupby("UserID", sort=False):
        latest_df, meta = find_latest_session_from_tail(
            g, L=L, k=k, min_abs=min_abs, ratio=ratio, delta=0.0, relax=relax
        )
        part = latest_df[latest_df["InLatestSession"]].copy()
        if not part.empty:
            latest_parts.append(part)
            meta["UserID"] = uid
            metas.append(meta)

    latest_sessions_df = pd.concat(latest_parts, ignore_index=True) if latest_parts else pd.DataFrame()
    meta_df = pd.DataFrame(metas)
    return latest_sessions_df.sort_values(["UserID", "Datetime"]).reset_index(drop=True), meta_df


def build_loo_datasets_with_time(
    latest_df: pd.DataFrame,
    source_df_for_items: pd.DataFrame,
    max_len: int = 50,
    make_validation: bool = True,
    n_neg_valid: int = 99,
    n_neg_test: int = 99,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    del max_len  # kept for API compatibility with the notebook
    all_items = source_df_for_items["MovieID"].unique().astype(int).tolist()
    all_items_set = set(all_items)
    train_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    def sample_negatives(rng: np.random.Generator, user_items: set[int], n: int) -> list[int]:
        pool = list(all_items_set - user_items)
        if not pool:
            return []
        idx = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
        return [int(pool[i]) for i in idx]

    for uid, g in latest_df.groupby("UserID", sort=False):
        g = g.sort_values("Datetime")
        items = g["MovieID"].astype(int).to_numpy()
        ts_sec = (g["Datetime"].astype("int64") // 1_000_000_000).to_numpy(np.int64)
        if len(items) < 2:
            continue

        gaps = np.diff(ts_sec, prepend=ts_sec[0]).astype(np.int64)
        test_idx = len(items) - 1
        valid_idx = len(items) - 2 if (make_validation and len(items) >= 3) else None

        end_train = valid_idx if valid_idx is not None else test_idx
        train_items, train_ts, train_gaps = items[:end_train], ts_sec[:end_train], gaps[:end_train]
        for t in range(1, len(train_items)):
            train_rows.append(
                {
                    "UserID": uid,
                    "hist": train_items[:t].tolist(),
                    "hist_ts": train_ts[:t].tolist(),
                    "hist_gap_prev": train_gaps[:t].tolist(),
                    "hist_age_to_target": (train_ts[t] - train_ts[:t]).astype(np.int64).tolist(),
                    "target": int(train_items[t]),
                    "target_ts": int(train_ts[t]),
                    "target_gap_prev": int(train_ts[t] - train_ts[t - 1]),
                    "hist_len": int(t),
                }
            )

        if valid_idx is not None:
            valid_rows.append(
                {
                    "UserID": uid,
                    "hist": items[:valid_idx].tolist(),
                    "hist_ts": ts_sec[:valid_idx].tolist(),
                    "hist_gap_prev": gaps[:valid_idx].tolist(),
                    "hist_age_to_target": (ts_sec[valid_idx] - ts_sec[:valid_idx]).astype(np.int64).tolist(),
                    "target": int(items[valid_idx]),
                    "target_ts": int(ts_sec[valid_idx]),
                    "target_gap_prev": int(ts_sec[valid_idx] - ts_sec[valid_idx - 1]),
                    "neg_items": sample_negatives(np.random.default_rng(seed + uid), set(items), n_neg_valid),
                    "hist_len": int(valid_idx),
                }
            )

        test_rows.append(
            {
                "UserID": uid,
                "hist": items[:test_idx].tolist(),
                "hist_ts": ts_sec[:test_idx].tolist(),
                "hist_gap_prev": gaps[:test_idx].tolist(),
                "hist_age_to_target": (ts_sec[test_idx] - ts_sec[:test_idx]).astype(np.int64).tolist(),
                "target": int(items[test_idx]),
                "target_ts": int(ts_sec[test_idx]),
                "target_gap_prev": int(ts_sec[test_idx] - ts_sec[test_idx - 1]),
                "neg_items": sample_negatives(np.random.default_rng(seed + uid + 1), set(items), n_neg_test),
                "hist_len": int(test_idx),
            }
        )

    return {
        "train": pd.DataFrame(train_rows),
        "valid": pd.DataFrame(valid_rows) if make_validation else pd.DataFrame(),
        "test": pd.DataFrame(test_rows),
    }


def pad_sequences_time(
    df_split: pd.DataFrame,
    max_len: int = 50,
    pad_item: int = 0,
) -> dict[str, torch.Tensor | None]:
    N = len(df_split)
    items = np.full((N, max_len), pad_item, dtype=np.int64)
    ts = np.zeros((N, max_len), dtype=np.int64)
    gap = np.zeros((N, max_len), dtype=np.int64)
    age2t = np.zeros((N, max_len), dtype=np.int64)
    lens = np.zeros(N, dtype=np.int64)

    for i, r in enumerate(df_split.itertuples(index=False)):
        h_items = list(r.hist)[-max_len:]
        h_ts = list(r.hist_ts)[-max_len:]
        h_gap = list(r.hist_gap_prev)[-max_len:]
        h_age = list(r.hist_age_to_target)[-max_len:]
        L = len(h_items)
        lens[i] = L
        if L == 0:
            continue
        items[i, -L:] = h_items
        ts[i, -L:] = h_ts
        gap[i, -L:] = h_gap
        age2t[i, -L:] = h_age

    neg_items = None
    if "neg_items" in df_split.columns and len(df_split) > 0:
        neg_items = torch.tensor(np.stack(df_split["neg_items"].values))

    return {
        "items": torch.tensor(items),
        "ts": torch.tensor(ts),
        "gap": torch.tensor(gap),
        "age2t": torch.tensor(age2t),
        "len": torch.tensor(lens),
        "target": torch.tensor(df_split["target"].to_numpy(np.int64)),
        "target_ts": torch.tensor(df_split.get("target_ts", pd.Series([0] * N)).to_numpy(np.int64)),
        "target_gap_prev": torch.tensor(df_split.get("target_gap_prev", pd.Series([0] * N)).to_numpy(np.int64)),
        "neg_items": neg_items,
    }
