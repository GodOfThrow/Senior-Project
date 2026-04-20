from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cop_recsys.data import (
    build_loo_datasets_with_time,
    extract_latest_sessions,
    load_movielens_ratings,
    pad_sequences_time,
)
from cop_recsys.datasets import create_session_dataloaders
from cop_recsys.models import COPContrastive, GRU4RecBaseline
from cop_recsys.training import train_model


def main():
    parser = argparse.ArgumentParser(description="Run cosine ablation for COP vs GRU4Rec.")
    parser.add_argument("--ratings-path", type=str, default="data/ratings.dat")
    parser.add_argument("--max-len", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print("Loading and preparing data...")
    ratings_df = load_movielens_ratings(args.ratings_path)
    latest_sessions_df, _ = extract_latest_sessions(
        ratings_df, L=10, k=3.0, min_abs=60, ratio=1.25, relax=True
    )
    ds_time = build_loo_datasets_with_time(
        latest_sessions_df,
        ratings_df,
        max_len=args.max_len,
        make_validation=True,
        seed=42,
    )

    train_data = pad_sequences_time(ds_time["train"], max_len=args.max_len)
    valid_data = pad_sequences_time(ds_time["valid"], max_len=args.max_len)
    test_data = pad_sequences_time(ds_time["test"], max_len=args.max_len)

    item_size = int(ratings_df["MovieID"].max() + 1)
    train_loader, valid_loader, test_loader = create_session_dataloaders(
        train_data,
        valid_data,
        test_data,
        num_items=item_size,
        batch_size=args.batch_size,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cop_decay = COPContrastive(item_size=item_size, d_model=128, use_time_decay=True)
    cop_decay_optim = torch.optim.Adam(cop_decay.parameters(), lr=args.lr, weight_decay=1e-5)
    train_model(
        cop_decay,
        train_loader,
        valid_loader,
        test_loader,
        cop_decay_optim,
        num_epochs=args.epochs,
        device=device,
        Ks=(5, 10, 20),
        lambda_contrastive=0.5,
        lambda_bpr=1.0,
        neighbor_metric="cosine",
        csv_filename="training_results_COPContrastive_Cosine_TimeDecay_FIXED_Lminus1.csv",
    )

    cop_no_decay = COPContrastive(item_size=item_size, d_model=128, use_time_decay=False)
    cop_no_decay_optim = torch.optim.Adam(cop_no_decay.parameters(), lr=args.lr, weight_decay=1e-5)
    train_model(
        cop_no_decay,
        train_loader,
        valid_loader,
        test_loader,
        cop_no_decay_optim,
        num_epochs=args.epochs,
        device=device,
        Ks=(5, 10, 20),
        lambda_contrastive=0.5,
        lambda_bpr=1.0,
        neighbor_metric="cosine",
        csv_filename="training_results_COPContrastive_Cosine_NoTimeDecay.csv",
    )

    gru_model = GRU4RecBaseline(num_items=item_size, emb_dim=128, hidden_dim=128)
    gru_optim = torch.optim.Adam(gru_model.parameters(), lr=args.lr)
    train_model(
        gru_model,
        train_loader,
        valid_loader,
        test_loader,
        gru_optim,
        num_epochs=args.epochs,
        device=device,
        Ks=(5, 10, 20),
        csv_filename="training_results_GRU4RecBaseline.csv",
    )


if __name__ == "__main__":
    main()
