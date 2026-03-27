"""Tach tap train va test."""

import pandas as pd
from sklearn.model_selection import train_test_split

# hàm separate_features_target giúp tách DataFrame thành features và target. 
# Hàm này đơn giản loại bỏ cột target khỏi DataFrame gốc để tạo ra features và 
# sau đó lấy cột target làm một Series riêng biệt. Việc tách features và target là 
# một bước quan trọng trong quá trình chuẩn bị dữ liệu cho mô hình học máy, 
# giúp chúng ta có thể dễ dàng áp dụng các bước xử lý riêng biệt cho features và target nếu cần thiết.
def separate_features_target(
    df: pd.DataFrame, 
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]: # trả về một tuple gồm DataFrame features và Series target
    """Tách DataFrame thành features và target."""
    features = df.drop(columns=[target_column]) # loại bỏ cột target khỏi DataFrame gốc để tạo ra features
    target = df[target_column]
    return features, target

# hàm split_features_target giúp tách DataFrame thành features và target,\
#  sau đó chia dữ liệu thành tập train và test dựa trên tỷ lệ test_size và random_state 
# được cung cấp. Hàm này là một bước quan trọng trong quá trình chuẩn bị dữ liệu cho mô hình
#  học máy, đảm bảo rằng chúng ta có một tập dữ liệu huấn luyện để xây dựng mô hình và một 
# tập dữ liệu kiểm tra để đánh giá hiệu suất của mô hình.
def split_features_target(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.3,
    random_state: int = 42,
):
    """Tách X, y và chia train/test."""
    features, target = separate_features_target(df, target_column)
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )

# hàm attach_target giúp ghép lại features với target dựa trên index hiện có, 
# đảm bảo rằng các bản ghi trong features và target được liên kết chính xác sau khi đã 
# được tách ra và có thể đã trải qua các bước xử lý riêng biệt.
def attach_target(
    features_df: pd.DataFrame,
    target_series: pd.Series,
    target_name: str = "Rings",
) -> pd.DataFrame:
    """Ghép lại features với target dựa trên index hiện có."""
    result = features_df.copy()
    result[target_name] = target_series.loc[features_df.index]
    return result
