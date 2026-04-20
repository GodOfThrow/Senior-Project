"""COP recommender package refactored from the project notebook."""

from .data import (
    build_loo_datasets_with_time,
    extract_latest_sessions,
    find_latest_session_from_tail,
    load_movielens_ratings,
    pad_sequences_time,
)
from .datasets import SessionDataset, create_session_dataloaders
from .training import evaluate, get_scores, summarize_best_epoch, train_model
from .models.cop_contrastive import COPContrastive
from .models.gru4rec import GRU4RecBaseline

__all__ = [
    "build_loo_datasets_with_time",
    "extract_latest_sessions",
    "find_latest_session_from_tail",
    "load_movielens_ratings",
    "pad_sequences_time",
    "SessionDataset",
    "create_session_dataloaders",
    "evaluate",
    "get_scores",
    "summarize_best_epoch",
    "train_model",
    "COPContrastive",
    "GRU4RecBaseline",
]
