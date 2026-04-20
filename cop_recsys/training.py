from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .models.cop_contrastive import COPContrastive
from .models.gru4rec import GRU4RecBaseline


def _get_batch_items_timestamps_target(batch: dict[str, torch.Tensor], device: str | torch.device):
    items = batch.get("items", batch.get("input_seq")).to(device)
    timestamps = batch.get("ts", batch.get("time_seq"))
    if timestamps is None:
        timestamps = torch.zeros_like(items)
    timestamps = timestamps.to(device)
    target = batch["target"].to(device)
    return items, timestamps, target


def _infer_num_items(model: nn.Module) -> int:
    if hasattr(model, "num_items"):
        return int(model.num_items)
    if hasattr(model, "enc") and hasattr(model.enc, "item_emb"):
        return int(model.enc.item_emb.num_embeddings)
    raise AttributeError("Cannot infer num_items from model.")


@torch.no_grad()
def get_scores(
    model: nn.Module,
    items: torch.Tensor,
    timestamps: torch.Tensor,
    target_item: torch.Tensor,
    neg_items: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    model.eval()
    if isinstance(model, GRU4RecBaseline):
        query_rep = model.encode_only(items)
        item_emb_table = model.enc.item_emb
        bias_table = model.item_bias
    else:
        enc_out = model.encode_only(items, timestamps)
        query_rep = enc_out["cls"]
        item_emb_table = model.enc.item_emb
        bias_table = None

    pos = target_item.unsqueeze(1)
    candidates = torch.cat([pos, neg_items], dim=1)
    cand_emb = item_emb_table(candidates)
    scores = (query_rep.unsqueeze(1) * cand_emb).sum(dim=-1)

    if bias_table is not None:
        scores += bias_table[candidates]
    return scores


def evaluate(
    model: nn.Module,
    dataloader,
    device: str | torch.device,
    Ks: list[int] | tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    model.eval()
    metrics = {f"HR@{k}": 0.0 for k in Ks}
    metrics.update({f"NDCG@{k}": 0.0 for k in Ks})

    count = 0
    for batch in dataloader:
        items, timestamps, target = _get_batch_items_timestamps_target(batch, device)
        neg_items = batch["neg_items"].to(device)
        scores = get_scores(model, items, timestamps, target, neg_items, device)
        _, indices = torch.sort(scores, descending=True, dim=-1)
        rank = (indices == 0).nonzero(as_tuple=False)[:, 1]

        for K in Ks:
            hit = (rank < K).float()
            metrics[f"HR@{K}"] += hit.sum().item()
            ndcg = hit * (1.0 / torch.log2(rank.float() + 2.0))
            metrics[f"NDCG@{K}"] += ndcg.sum().item()

        count += items.size(0)

    for k in metrics:
        metrics[k] /= max(count, 1)

    return metrics


def summarize_best_epoch(
    history: dict[str, list[float]],
    model_name: str,
    neighbor_metric: str = "-",
    use_time_decay: Any = "-",
) -> dict[str, Any]:
    rank_key = "test_NDCG@10" if "test_NDCG@10" in history else next(
        k for k in history.keys() if k.startswith("test_NDCG@")
    )
    best_idx = int(np.argmax(history[rank_key]))
    row: dict[str, Any] = {
        "Model": model_name,
        "BestEpoch": best_idx + 1,
        "TrainLoss": history["train_loss"][best_idx],
        "NeighborMetric": neighbor_metric,
        "UseTimeDecay": use_time_decay,
    }
    for key, values in history.items():
        if key.startswith("test_"):
            row[key.replace("test_", "")] = values[best_idx]
    return row


def train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    test_loader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    device: str | torch.device = "cpu",
    Ks: tuple[int, ...] = (5, 10, 20),
    lambda_contrastive: float = 0.5,
    lambda_bpr: float = 1.0,
    neighbor_metric: str = "l2",
    csv_filename: str | None = None,
    verbose_debug: bool = True,
) -> dict[str, list[float]]:
    del valid_loader
    model.to(device)
    is_gru = isinstance(model, GRU4RecBaseline)
    criterion_ce = nn.CrossEntropyLoss()
    num_items = _infer_num_items(model)

    history: dict[str, list[float]] = {"train_loss": []}
    for k in Ks:
        history[f"test_HR@{k}"] = []
        history[f"test_NDCG@{k}"] = []

    export_logs: list[dict[str, Any]] = []
    use_time_decay = getattr(model, "use_time_decay", None)

    print(f"Start Training {type(model).__name__}...")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        epoch_debug_vals: dict[str, Any] = {}

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{num_epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()
            items, timestamps, target = _get_batch_items_timestamps_target(batch, device)

            if is_gru:
                logits = model.full_scores(items)
                loss = criterion_ce(logits, target)
            else:
                out = model(
                    items=items,
                    timestamps=timestamps,
                    bank_is_batch=True,
                    topk=5,
                    num_negatives=5,
                    floor=1e-3,
                    predict_mode="linear",
                    metric=neighbor_metric,
                    return_loss=True,
                )
                loss_ctr = out["loss"]

                if "neg_item" in batch:
                    neg_item = batch["neg_item"].to(device)
                else:
                    neg_item = torch.randint(1, num_items, size=target.shape, device=device)
                    neg_item = torch.where(
                        neg_item == target,
                        (neg_item % (num_items - 1)) + 1,
                        neg_item,
                    )

                cls_anchor = out["cls_anchor"]
                item_emb = model.enc.item_emb
                e_pos = item_emb(target)
                e_neg = item_emb(neg_item)
                s_pos = (cls_anchor * e_pos).sum(-1)
                s_neg = (cls_anchor * e_neg).sum(-1)
                loss_bpr_val = -F.logsigmoid(s_pos - s_neg).mean()

                loss = (lambda_contrastive * loss_ctr) + (lambda_bpr * loss_bpr_val)

            if batch_idx == 0 and (not is_gru) and verbose_debug:
                print(f"\n{'=' * 20} DEBUG EPOCH {epoch} (Sample 0) {'=' * 20}")
                with torch.no_grad():
                    embeddings = out["pairs_anchor"]
                    print("1. Matrix Before Neighbor Calc (The Consecutive Pair Features):")
                    print(f"   Shape: {list(embeddings.shape)}  [Batch, Pair_Len, Dim]")

                    full_vec = embeddings[0, -1, :]
                    avg_val = full_vec.mean().item()
                    std_val = full_vec.std().item()
                    abs_avg = full_vec.abs().mean().item()

                    sample_vec_8 = full_vec[:8].detach().cpu().numpy()
                    print("   Sample Vector (First 8 dims):")
                    print(f"   [{', '.join([f'{x:.4f}' for x in sample_vec_8])} ...]")
                    print(f"   >> Stat Summary: Mean={avg_val:.6f}, Std={std_val:.6f}, Abs_Mean={abs_avg:.6f}")
                    print(f"   >> Anchor-Only Time Decay Enabled: {getattr(model, 'use_time_decay', False)}")

                    dists = None
                    sims = None
                    neighbor_info: dict[str, Any] = {}

                    print("\n2. Checking Neighbors (Top-5 Nearest):")
                    if neighbor_metric.lower() == "l2":
                        dists = out["topk_dist"][0].detach().cpu().numpy() if out.get("topk_dist") is not None else None
                        sims = out["topk_sim"][0].detach().cpu().numpy() if out.get("topk_sim") is not None else None
                        print("Metric: L2")
                        if dists is not None:
                            print("Top-k distances:", dists)
                            neighbor_info["Distances (L2)"] = str(dists.tolist())
                        if sims is not None:
                            print("Converted similarity weights:", sims)
                            neighbor_info["Similarity Weights"] = str(sims.tolist())
                    elif neighbor_metric.lower() in ("cos", "cosine", "cosine_similarity"):
                        sims = out["topk_sim"][0].detach().cpu().numpy() if out.get("topk_sim") is not None else None
                        print("Metric: Cosine")
                        if sims is not None:
                            print("Top-k cosine weights:", sims)
                            neighbor_info["Cosine Weights"] = str(sims.tolist())

                    l_pos = out["logs"]["logits_pos"][0].detach().cpu().numpy()
                    l_neg = out["logs"]["logits_neg"][0].detach().cpu().numpy()
                    print("\n3. InfoNCE Logits:")
                    print(f"   Pos Logits    : {np.round(l_pos, 2)}")
                    print(f"   Neg Logits    : {np.round(l_neg, 2)}")
                    print(f"{'=' * 60}\n")

                    epoch_debug_vals = {
                        "Sample Vector (First 8 dims)": str(sample_vec_8.tolist()),
                        "Mean": avg_val,
                        "Std": std_val,
                        "Abs_Mean": abs_avg,
                        "NeighborMetric": neighbor_metric,
                        "UseTimeDecay": getattr(model, "use_time_decay", False),
                        "Pos Logits": str(l_pos.tolist()),
                        "Neg Logits": str(l_neg.tolist()),
                    }
                    epoch_debug_vals.update(neighbor_info)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / max(steps, 1)
        history["train_loss"].append(avg_loss)

        test_metrics = evaluate(model, test_loader, device, Ks=Ks)
        current_row: dict[str, Any] = {
            "Epoch": epoch,
            "Loss": avg_loss,
            "NeighborMetric": "-" if is_gru else neighbor_metric,
            "UseTimeDecay": "-" if is_gru else bool(use_time_decay),
        }
        for k in Ks:
            current_row[f"HR@{k}"] = test_metrics[f"HR@{k}"]
            current_row[f"NDCG@{k}"] = test_metrics[f"NDCG@{k}"]

        current_row.update(epoch_debug_vals)
        export_logs.append(current_row)

        for k in Ks:
            history[f"test_HR@{k}"].append(test_metrics[f"HR@{k}"])
            history[f"test_NDCG@{k}"].append(test_metrics[f"NDCG@{k}"])

        print(
            f"Ep {epoch} | Loss: {avg_loss:.4f} | "
            f"HR@10: {test_metrics['HR@10']:.4f} NDCG@10: {test_metrics['NDCG@10']:.4f}"
        )

    df_export = pd.DataFrame(export_logs)
    if csv_filename is None:
        csv_filename = f"training_results_{type(model).__name__}.csv"
    df_export.to_csv(csv_filename, index=False)
    print(f"\nSaved: {csv_filename}")
    return history
