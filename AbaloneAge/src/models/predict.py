"""Suy luận trên dữ liệu mới."""


def predict(model, x_input):
    """Trả về kết quả dự đoán từ mô hình đã huấn luyện."""
    return model.predict(x_input)
