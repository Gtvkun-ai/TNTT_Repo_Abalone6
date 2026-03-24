"""Tien ich dung chung trong du an."""

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Tao thu muc neu chua ton tai va tra ve doi tuong Path."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
