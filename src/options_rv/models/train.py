"""Model training utilities for compact linear forecasting models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RidgeTrainingResult:
    """Container with fitted model outputs for evaluation and interpretation.

    Parameters
    ----------
    model
        Fitted scikit-learn pipeline with standardization and ridge regressor.
    predictions
        Prediction table with original keys and fitted values.
    coefficients
        Coefficient table in standardized feature space.
    """

    model: Pipeline
    predictions: pd.DataFrame
    coefficients: pd.DataFrame


def train_ridge_regression(
    feature_target_frame: pd.DataFrame,
    split_labels: pd.Series,
    feature_columns: list[str],
    target_column: str,
    ridge_alpha: float = 1.0,
) -> RidgeTrainingResult:
    """Fit ridge regression on train split and predict across all rows.

    Parameters
    ----------
    feature_target_frame
        Feature-target panel with ``symbol`` and ``trade_date`` keys.
    split_labels
        Label vector with values ``train``, ``validation``, ``test``.
    feature_columns
        Ordered list of feature columns used by the model.
    target_column
        Target variable column.
    ridge_alpha
        L2 penalty weight used by ridge regression.

    Returns
    -------
    RidgeTrainingResult
        Model object, prediction table, and coefficient table.
    """
    model_columns = feature_columns + [target_column]
    missing_columns = [
        column for column in model_columns if column not in feature_target_frame.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Model columns are missing: {', '.join(missing_columns)}"
        )

    # Keep only rows where every model input and the target are observed.
    working_frame = feature_target_frame.copy()
    complete_row_mask = working_frame[model_columns].notna().all(axis=1)
    working_frame = working_frame.loc[complete_row_mask].reset_index(drop=True)
    split_labels = split_labels.loc[complete_row_mask].reset_index(drop=True)

    train_mask = split_labels.eq("train")
    if not train_mask.any():
        raise ValueError("split_labels does not include any train rows")

    x_train = working_frame.loc[train_mask, feature_columns]
    y_train = working_frame.loc[train_mask, target_column]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=ridge_alpha, random_state=0)),
        ]
    )
    model.fit(x_train, y_train)

    fitted_values = model.predict(working_frame[feature_columns])
    prediction_frame = working_frame.loc[
        :, ["symbol", "trade_date", target_column]
    ].copy()
    prediction_frame["split"] = split_labels
    prediction_frame["prediction"] = fitted_values

    ridge_step: Ridge = model.named_steps["ridge"]
    coefficient_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": ridge_step.coef_,
        }
    )

    return RidgeTrainingResult(
        model=model,
        predictions=prediction_frame,
        coefficients=coefficient_frame,
    )
