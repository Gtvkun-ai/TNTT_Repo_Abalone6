"""So sánh các cách tiền xử lý cho bài toán hồi quy Abalone."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from _common import RANDOM_STATE, RAW_FILE, TEST_SIZE, TRAIN_SIZE, save_results
from src.data.load_data import load_abalone_data
from src.data.split_data import split_features_target
from src.features.feature_engineering import (
    build_encoded_preprocessor,
    build_robust_scaled_preprocessor,
    build_standard_scaled_preprocessor,
    prepare_abalone_feature_groups,
    transform_with_preprocessor,
)
from src.models.evaluate import evaluate_regression_model

# evaluate_pipeline là một hàm tiện ích để đánh giá một cặp tiền xử lý và mô hình 
# bằng cách thực hiện các bước chuẩn bị dữ liệu, huấn luyện mô hình và tính toán các chỉ số đánh giá.
# Điều này giúp giảm sự trùng lặp mã và làm cho quá trình đánh giá các cặp tiền xử lý 
# và mô hình trở nên dễ dàng và nhất quán hơn trong toàn bộ dự án.
def evaluate_pipeline(preprocessor, model, x_train, x_test, y_train, y_test) -> dict:
    """Fit, predict va tinh metric cho mot cap preprocess-model."""
    x_train_ready, x_test_ready = transform_with_preprocessor(
        preprocessor,
        x_train,
        x_test,
    )
    model.fit(x_train_ready, y_train)
    predictions = model.predict(x_test_ready) 
    return evaluate_regression_model(y_test, predictions)

# run_experiment là hàm chính để chạy thí nghiệm so sánh các phương pháp tiền xử lý khác nhau trên tập dữ liệu Abalone,
#  sử dụng cả Linear Regression và Random Forest Regressor để đánh giá hiệu suất của từng phương
#  pháp tiền xử lý. Kết quả được lưu trữ trong một cấu trúc dữ liệu có tổ chức để dễ dàng phân tích và so sánh sau này.
def run_experiment() -> dict:
    """chạy so sánh preprocessing."""
    df = load_abalone_data(RAW_FILE)
    groups = prepare_abalone_feature_groups()
    categorical_cols = list(groups["categorical_cols"])
    numeric_cols = list(groups["numeric_cols"])
    target_col = str(groups["target_col"])

    x_train, x_test, y_train, y_test = split_features_target(
        df,
        target_column=target_col,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    # Định nghĩa các cặp tiền xử lý và mô hình để đánh giá trong thí nghiệm so sánh.
    # Các cặp này bao gồm:
    # - Linear Regression với One-Hot Encoding cho các đặc trưng phân loại.
    # - Linear Regression với Standard Scaling cho các đặc trưng số và One-Hot Encoding cho các đặc trưng phân loại.
    # - Random Forest Regressor với Robust Scaling cho các đặc trưng số và One-Hot Encoding cho các đặc trưng phân loại.
    results = {
        "linear_encoded_only": evaluate_pipeline(
            build_encoded_preprocessor(categorical_cols),
            LinearRegression(),
            x_train,
            x_test,
            y_train,
            y_test,
        ),
        "linear_standard_scaled": evaluate_pipeline(
            build_standard_scaled_preprocessor(categorical_cols, numeric_cols),
            LinearRegression(),
            x_train,
            x_test,
            y_train,
            y_test,
        ),
        "rf_robust_scaled": evaluate_pipeline(
            build_robust_scaled_preprocessor(categorical_cols, numeric_cols),
            RandomForestRegressor(
                n_estimators=250,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            x_train,
            x_test,
            y_train,
            y_test,
        ),
    }

    return {
        "experiment_name": "preprocessing_comparison",
        "dataset": "abalone",
        "train_size": TRAIN_SIZE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "results": results,
    }


if __name__ == "__main__":
    results = run_experiment()
    exp_path, out_path = save_results("02_preprocessing_comparison.json", results)
    print(f"Saved results to: {exp_path}")
    print(f"Saved results to: {out_path}")
