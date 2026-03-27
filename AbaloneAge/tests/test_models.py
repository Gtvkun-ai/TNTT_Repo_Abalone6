import math

from src.models.evaluate import calculate_rse, evaluate_regression_metrics, evaluate_regression_model
# Các hàm trong module này tập trung vào việc kiểm tra các chức năng đánh giá mô hình hồi quy,
#  giúp đảm bảo rằng các metric được tính toán chính xác và phản ánh đúng hiệu suất của mô hình. 
# Chúng giúp đảm bảo rằng các metric như MSE, RMSE, RSE, MAE và R² được tính toán đúng cách, 
# từ đó cung cấp thông tin chính xác về hiệu suất của mô hình hồi quy và hỗ trợ trong việc 
# cải thiện mô hình hoặc lựa chọn mô hình phù hợp cho bài toán cụ thể.

def test_evaluate_regression_model_returns_metrics():
    metrics = evaluate_regression_model([1, 2, 3], [1, 2, 3])

    assert set(metrics.keys()) == {"mae", "mse", "rmse", "rse", "r2"}
    assert metrics["mae"] == 0
    assert metrics["mse"] == 0
    assert metrics["rmse"] == 0
    assert metrics["r2"] == 1


def test_calculate_rse_matches_expected_ratio():
    rse = calculate_rse([1, 2, 3], [1, 1, 1])

    assert math.isclose(rse, 2.5)


def test_evaluate_regression_metrics_returns_project_metrics():
    metrics = evaluate_regression_metrics([1, 2, 3], [1, 2, 3])

    assert set(metrics.keys()) == {"mae", "mse", "rmse", "rse"}
