"""Xu ly dac trung cho mo hinh."""

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Noi dung feature engineering se duoc bo sung tai day."""
    return df.copy()
