"""So sanh du lieu goc va du lieu co bien doi feature engineering."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from _common import RANDOM_STATE, RAW_FILE, TEST_SIZE, TRAIN_SIZE, save_results
from src.data.load_data import load_abalone_data
from src.data.split_data import split_features_target
from src.features.feature_engineering import (
    build_encoded_preprocessor,
    prepare_abalone_feature_groups,
    transform_with_preprocessor,
)
from src.models.evaluate import evaluate_regression_model


def add_manual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo một số feature thủ công đơn giản để thử nghiệm."""
    enriched = df.copy()
    enriched["VolumeApprox"] = (
        enriched["Length"] * enriched["Diameter"] * enriched["Height"]
    )
    enriched["TotalWeightToShell"] = (
        enriched["WholeWeight"] / (enriched["ShellWeight"] + 1e-6)
    )
    enriched["EdibleWeightRatio"] = (
        (enriched["ShuckedWeight"] + enriched["VisceraWeight"])
        / (enriched["WholeWeight"] + 1e-6)
    )
    return enriched


def evaluate_dataset_variant(df: pd.DataFrame) -> dict:
    """Đánh giá một biến thể dữ liệu bằng RandomForest."""
    groups = prepare_abalone_feature_groups()
    target_col = str(groups["target_col"])
    categorical_cols = list(groups["categorical_cols"])

    x_train, x_test, y_train, y_test = split_features_target(
        df,
        target_column=target_col,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_encoded_preprocessor(categorical_cols)
    x_train_ready, x_test_ready = transform_with_preprocessor(
        preprocessor,
        x_train,
        x_test,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train_ready, y_train)
    predictions = model.predict(x_test_ready)
    return evaluate_regression_model(y_test, predictions)


def run_experiment() -> dict:
    """Chạy ablation study cho feature engineering."""
    base_df = load_abalone_data(RAW_FILE)
    enriched_df = add_manual_features(base_df)

    return {
        "experiment_name": "feature_engineering_ablation",
        "dataset": "abalone",
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "variants": {
            "raw_features": evaluate_dataset_variant(base_df),
            "manual_feature_engineering": evaluate_dataset_variant(enriched_df),
        },
        "added_features": [
            "VolumeApprox",
            "TotalWeightToShell",
            "EdibleWeightRatio",
        ],
    }


if __name__ == "__main__":
    results = run_experiment()
    exp_path, out_path = save_results("03_feature_engineering_ablation.json", results)
    print(f"Saved results to: {exp_path}")
    print(f"Saved results to: {out_path}")
