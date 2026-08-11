"""Baseline and linear model training helpers for options RV research."""

from .baselines import compute_atm_iv_baseline, compute_persistence_baseline
from .train import train_ridge_regression

__all__ = [
    "compute_atm_iv_baseline",
    "compute_persistence_baseline",
    "train_ridge_regression",
]
