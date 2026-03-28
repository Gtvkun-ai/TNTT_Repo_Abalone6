"""Tien ich dung chung cho cac script thu nghiem."""

from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import OUTPUT_DIR, RAW_DATA_DIR


EXPERIMENTS_DIR = Path(__file__).resolve().parent
EXPERIMENT_RESULTS_DIR = EXPERIMENTS_DIR / "results"
OUTPUT_EXPERIMENTS_DIR = OUTPUT_DIR / "metrics" / "experiments"
RAW_FILE = RAW_DATA_DIR / "abalone.csv"
RANDOM_STATE = 42
TRAIN_SIZE = 0.7
TEST_SIZE = 0.3
CV_SPLITS = 5


def ensure_result_dirs() -> None:
    """Tao cac thu muc can thiet de luu ket qua."""
    EXPERIMENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def save_results(file_name: str, payload: dict) -> tuple[Path, Path]:
    """Luu ket qua vao hai vi tri de de theo doi."""
    ensure_result_dirs()
    exp_path = EXPERIMENT_RESULTS_DIR / file_name
    out_path = OUTPUT_EXPERIMENTS_DIR / file_name

    serialized = json.dumps(payload, indent=2, ensure_ascii=True)
    exp_path.write_text(serialized + "\n", encoding="utf-8")
    out_path.write_text(serialized + "\n", encoding="utf-8")
    return exp_path, out_path
