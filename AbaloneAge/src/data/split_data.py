"""Tach tap train va test."""

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Tach X, y va chia train/test."""
    features = df.drop(columns=[target_column])
    target = df[target_column]
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )
