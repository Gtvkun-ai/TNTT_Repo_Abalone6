"""Danh gia mo hinh."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

# Hàm calculate_rse tính toán Relative Squared Error (RSE) để đánh giá hiệu suất của mô hình hồi quy, 
# giúp chúng ta hiểu được mức độ lỗi của mô hình so với độ biến thiên của dữ liệu thực tế. 
# RSE là một metric quan trọng để đánh giá chất lượng của mô hình hồi quy, 
# đặc biệt khi so sánh giữa các mô hình khác nhau. 
def calculate_rse(y_true, y_pred) -> float:
    """TÍnh Relative Squared Error."""
    y_true_array = np.asarray(y_true) # chuyển đổi y_true và y_pred thành mảng NumPy để đảm bảo rằng các phép toán số học có thể được thực hiện một cách hiệu quả và chính xác.
    y_pred_array = np.asarray(y_pred)
    numerator = np.sum((y_true_array - y_pred_array) ** 2) # tính tổng bình phương sai số giữa giá trị thực tế và giá trị dự đoán, giúp đánh giá mức độ lỗi của mô hình.
    denominator = np.sum((y_true_array - np.mean(y_true_array)) ** 2) # tính tổng bình phương sai số giữa giá trị thực tế và giá trị trung bình của giá trị thực tế, giúp đánh giá độ biến thiên của dữ liệu thực tế.
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)

# Hàm evaluate_regression_metrics tính toán một bộ các metric hồi quy như MSE, RMSE, RSE và MAE để đánh giá hiệu suất của mô hình hồi quy.
#  Việc sử dụng nhiều metric khác nhau giúp chúng ta có cái nhìn toàn diện hơn về hiệu suất của mô hình, từ đó có thể đưa ra quyết định tốt hơn về việc cải thiện mô hình hoặc lựa chọn mô hình phù hợp cho bài toán cụ thể.
def evaluate_regression_metrics(y_true, y_pred) -> dict:
    """Trả về bộ metric hồi quy dùng trong notebook."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "rse": calculate_rse(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }

# Hàm evaluate_regression_model mở rộng bộ metric hồi quy bằng cách thêm R² (R-squared) vào kết quả, 
# giúp đánh giá mức độ giải thích của mô hình đối với biến phụ thuộc. 
# R² là một metric quan trọng để đánh giá hiệu suất của mô hình hồi quy, 
# đặc biệt khi so sánh giữa các mô hình khác nhau.
def evaluate_regression_model(y_true, y_pred) -> dict:
    """Trả về đầy đủ metric hồi quy để sử dụng trong src và test."""
    metrics = evaluate_regression_metrics(y_true, y_pred)
    metrics["r2"] = r2_score(y_true, y_pred)
    return metrics

# Hàm build_kfold tạo một đối tượng KFold với cấu hình mặc định được sử dụng trong dự án,
#  giúp đảm bảo rằng các bước cross-validation được thực hiện một cách nhất quán và
#  dễ dàng tái sử dụng trong các phần khác nhau của dự án mà không cần phải cấu hình lại mỗi lần.
# KFlod là một chiến lược cross-validation phổ biến giúp đánh giá hiệu suất của mô hình 
# bằng cách chia dữ liệu thành K phần và sử dụng lần lượt mỗi phần làm tập kiểm tra 
# trong khi các phần còn lại làm tập huấn luyện.
def build_kfold(
    n_splits: int = 5,
    *,
    shuffle: bool = True,
    random_state: int = 42,
) -> KFold:
    """Tạo cấu hình KFold mặc định của dự án."""
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

# Hàm run_cv_predictions thực hiện cross-validation
#  để lấy dự đoán OOF (Out-Of-Fold) và tính toán các metric hồi quy dựa trên các dự đoán này.
#  Việc sử dụng cross-validation giúp đánh giá hiệu suất của mô hình một cách chính xác hơn 
# bằng cách sử dụng tất cả dữ liệu có sẵn cho cả huấn luyện và kiểm tra,
#  đồng thời giúp giảm thiểu nguy cơ overfitting và cung cấp cái nhìn tổng quan về hiệu suất của mô hình
#  trên các tập dữ liệu khác nhau.
def run_cv_predictions(model, x_data, y_data, cv_strategy: KFold):
    """Lấy dự đoán OOF và metric từ cross-validation."""
    y_oof_pred = cross_val_predict(model, x_data, y_data, cv=cv_strategy, n_jobs=-1) # sử dụng cross_val_predict để thực hiện cross-validation và lấy dự đoán OOF (Out-Of-Fold) cho tập dữ liệu, giúp đánh giá hiệu suất của mô hình một cách chính xác hơn.
    metrics = evaluate_regression_metrics(y_data, y_oof_pred) # tính toán các metric hồi quy dựa trên các dự đoán OOF, giúp đánh giá hiệu suất của mô hình trên toàn bộ tập dữ liệu.
    return y_oof_pred, metrics
