"""Evaluation metrics for forward-variance forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd


def root_mean_squared_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute root mean squared error (RMSE) in variance level space."""
    residual = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(residual**2)))


def mean_absolute_error(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute mean absolute error (MAE) in variance level space."""
    residual = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return float(np.mean(np.abs(residual)))


def qlike(
    actual_variance: pd.Series, predicted_variance: pd.Series, epsilon: float = 1e-12
) -> float:
    """Compute QLIKE loss in variance space.

    QLIKE = mean(log(predicted) + actual/predicted)
    """
    actual = actual_variance.to_numpy(dtype=float)
    # Floor predicted variance so log and ratio terms stay finite.
    predicted = np.maximum(predicted_variance.to_numpy(dtype=float), epsilon)
    return float(np.mean(np.log(predicted) + (actual / predicted)))
