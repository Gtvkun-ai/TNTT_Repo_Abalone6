"""Thu nghiem baseline voi RandomForestRegressor."""

from __future__ import annotations

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
from src.models.train import train_baseline_model


def run_experiment() -> dict:
    """chạy baseline và trả về metric."""
    df = load_abalone_data(RAW_FILE)
    groups = prepare_abalone_feature_groups()
    target_col = str(groups["target_col"])
    categorical_cols = list(groups["categorical_cols"])

    x_train, x_test, y_train, y_test = split_features_target(
        df,
        target_column=target_col,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # build encoded preprocessor là bước tiền xử lý cơ bản để mã hóa các đặc trưng phân loại, đảm bảo rằng mô hình có thể xử lý dữ liệu một cách hiệu quả.
    preprocessor = build_encoded_preprocessor(categorical_cols)
    x_train_ready, x_test_ready = transform_with_preprocessor(
        preprocessor,
        x_train,
        x_test,
    )

    
    baseline_model = train_baseline_model(x_train_ready, y_train)
    baseline_predictions = baseline_model.predict(x_test_ready)
    baseline_metrics = evaluate_regression_model(y_test, baseline_predictions)

    # Tuning một số siêu tham số của RandomForestRegressor để cải thiện hiệu suất so với baseline.
    # Việc này giúp đánh giá xem việc điều chỉnh mô hình có mang lại lợi ích đáng kể hay không so với cấu hình mặc định.
    # Các siêu tham số được điều chỉnh bao gồm số lượng cây (n_estimators), độ sâu tối đa của cây (max_depth) và số mẫu tối thiểu trong một lá (min_samples_leaf).
    tuned_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=RANDOM_STATE, # Đảm bảo tính tái lập của kết quả bằng cách sử dụng một giá trị ngẫu nhiên cố định.
        n_jobs=-1, # Sử dụng tất cả các lõi CPU để tăng tốc quá trình huấn luyện.
    )
    tuned_model.fit(x_train_ready, y_train)
    tuned_predictions = tuned_model.predict(x_test_ready)
    tuned_metrics = evaluate_regression_model(y_test, tuned_predictions)

    return {
        "experiment_name": "baseline_random_forest",
        "dataset": "abalone",
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "models": {
            "baseline_random_forest": baseline_metrics,
            "tuned_random_forest": tuned_metrics,
        },
    }


if __name__ == "__main__":
    results = run_experiment()
    exp_path, out_path = save_results("01_baseline_random_forest.json", results)
    print(f"Saved results to: {exp_path}")
    print(f"Saved results to: {out_path}")
