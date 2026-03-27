"""Ham doc du lieu goc va du lieu Abalone."""

from pathlib import Path
from typing import Any

import pandas as pd


ABALONE_COLUMNS = [
    "Sex",
    "Length",
    "Diameter",
    "Height",
    "WholeWeight",
    "ShuckedWeight",
    "VisceraWeight",
    "ShellWeight",
    "Rings",
]


def load_csv(file_path: str | Path, **read_csv_kwargs: Any) -> pd.DataFrame:
    """Đọc file CSV."""
    return pd.read_csv(file_path, **read_csv_kwargs)


def load_abalone_data(
    file_path: str | Path,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Đọc file Abalone gốc với tên cột đã cho."""
    resolved_columns = columns or ABALONE_COLUMNS
    return load_csv(file_path, header=None, names=resolved_columns)
