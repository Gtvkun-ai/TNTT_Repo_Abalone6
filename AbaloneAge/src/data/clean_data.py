"""Cac ham tien xu ly va lam sach du lieu."""

import pandas as pd


def clean_abalone_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tra ve ban sao du lieu sau khi lam sach co ban."""
    cleaned_df = df.copy()
    cleaned_df.columns = [column.strip().lower() for column in cleaned_df.columns]
    return cleaned_df
