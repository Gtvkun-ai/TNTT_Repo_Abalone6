"""Ve bieu do ket qua."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Các hàm trong module này tập trung vào việc trực quan hóa kết quả của mô hình hồi quy,
#  bao gồm việc so sánh giá trị thực và dự đoán, phân tích residuals, so sánh các metric giữa các mô hình khác nhau và vẽ learning curve.


# Hàm plot_prediction_scatter vẽ biểu đồ so sánh giữa giá trị thực và 
# giá trị dự đoán của mô hình hồi quy, giúp đánh giá trực quan hiệu suất của mô hình.
#  Việc so sánh trực quan này rất hữu ích để nhận biết các mẫu lỗi phổ biến, 
# xác định xem mô hình có xu hướng dự đoán quá cao hoặc quá thấp hay không, và đánh giá mức độ phù hợp của mô hình với dữ liệu thực tế.
def plot_prediction_scatter(y_true, y_pred, model_name: str | None = None, ax=None):
    """Vẽ biểu đồ so sánh giá trị thực và dự đoán."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, ax=ax) # sử dụng seaborn để vẽ biểu đồ scatter giữa giá trị thực và giá trị dự đoán, giúp đánh giá trực quan hiệu suất của mô hình hồi quy.
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Gia tri thuc")
    ax.set_ylabel("Gia tri du doan")
    title = "So sanh gia tri thuc va du doan"
    if model_name:
        title = f"Actual vs Predicted - {model_name}"
    ax.set_title(title)
    return fig, ax

# Hàm plot_residual_scatter vẽ biểu đồ residuals để phân tích sai số giữa giá trị thực 
# và giá trị dự đoán của mô hình hồi quy, giúp đánh giá xem mô hình có phù hợp với dữ liệu hay không.
#  Việc phân tích residuals rất quan trọng để nhận biết các mẫu lỗi, xác định xem mô hình 
# có xu hướng dự đoán quá cao hoặc quá thấp hay không, và đánh giá mức độ phù hợp của mô hình 
# với dữ liệu thực tế. Nếu residuals có phân phối ngẫu nhiên xung quanh đường ngang tại 0,
#  điều này thường cho thấy mô hình phù hợp tốt với dữ liệu. Ngược lại, nếu residuals có mẫu cụ thể
#  hoặc xu hướng, điều này có thể chỉ ra rằng mô hình không phù hợp hoặc có vấn đề với dữ liệu.
def plot_residual_scatter(y_true, y_pred, model_name: str | None = None, ax=None):
    """Vẽ residual plot cho bài toán hồi quy."""
    if ax is None:
        fig, ax = plt.subplots()  # tạo một figure và axes mới nếu ax không được cung cấp
    else:
        fig = ax.figure

    residuals = np.asarray(y_true) - np.asarray(y_pred) # tính toán residuals bằng cách lấy hiệu giữa giá trị thực và giá trị dự đoán, giúp đánh giá sai số của mô hình.
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6, ax=ax)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Giá trị dự đoán")
    ax.set_ylabel("Sai số (y_true - y_pred)")
    title = "Residual Plot"
    if model_name:
        title = f"Residual Plot - {model_name}"
    ax.set_title(title)
    return fig, ax


# Hàm plot_metric_comparison_grid vẽ lưới biểu đồ so sánh các metric giữa các mô hình khác nhau,
#  giúp đánh giá và so sánh hiệu suất của các mô hình một cách trực quan.
#  Việc so sánh trực quan này rất hữu ích để nhận biết mô hình nào hoạt động tốt hơn dựa trên 
# các metric khác nhau, từ đó có thể đưa ra quyết định về việc cải thiện mô hình hoặc 
# lựa chọn mô hình phù hợp cho bài toán cụ thể.
def plot_metric_comparison_grid(
    df,
    metric_columns: list[str],
    *,
    model_column: str = "model",
    figsize: tuple[int, int] = (16, 10),
):
    """Vẽ lưới bar chart cho nhiều metric trên cùng một bảng kết quả."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, metric in zip(axes.flat, metric_columns):
        sns.barplot(data=df, x=metric, y=model_column, ax=ax)
        ax.set_title(f"So sanh {metric}")
    plt.tight_layout()
    return fig, axes

# Hàm plot_learning_curve_lines vẽ biểu đồ learning curve từ các giá trị train size và hai dãy score (train và validation),
#  giúp đánh giá hiệu suất của mô hình theo kích thước tập huấn luyện và xác định 
# xem mô hình có bị overfitting hay underfitting hay không.
#  Việc vẽ learning curve giúp chúng ta hiểu rõ hơn về cách mô hình học từ dữ liệu và 
# có thể cung cấp thông tin quan trọng để điều chỉnh mô hình hoặc thu thập thêm dữ liệu nếu cần thiết.  
def plot_learning_curve_lines(
    train_sizes,
    train_scores,
    validation_scores,
    *,
    title: str,
    ax=None,
):
    """Vẽ learning curve từ train size và hai dãy score."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(train_sizes, train_scores, marker="o", label="Train")
    ax.plot(train_sizes, validation_scores, marker="s", label="Validation")
    ax.set_xlabel("Số mẫu huấn luyện")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig, ax
