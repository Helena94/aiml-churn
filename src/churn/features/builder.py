# src/churn/features/builder.py

import pandas as pd
from churn.features.schema import ID_COLUMNS, TARGET_COL, ID_COLS, TARGET_COLUMN


def split_features_target(df: pd.DataFrame):
    """
    Returns X, y (raw, unencoded).
    """
    X = df.drop(columns=TARGET_COLUMN + ID_COLUMNS)
    y = df[TARGET_COLUMN]

    return X, y


def build_features(df: pd.DataFrame):
    """
    Used for inference (no y).
    """
    X = df.drop(columns=ID_COLUMNS + TARGET_COLUMN, errors="ignore")
    return X
