"""Ham doc du lieu goc."""

from pathlib import Path

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Doc file CSV va tra ve DataFrame."""
    return pd.read_csv(file_path)
