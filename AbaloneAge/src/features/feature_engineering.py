"""Xu ly dac trung va helper tien xu ly cho mo hinh."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

# Các hàm trong module này tập trung vào việc xây dựng các đặc trưng mới, 
# tạo các bộ tiền xử lý cho dữ liệu, và áp dụng các biến đổi như log1p để 
# giảm độ lệch của dữ liệu số. Chúng giúp chuẩn bị dữ liệu một cách hiệu quả trước 
# khi đưa vào mô hình học máy, đảm bảo rằng các đặc trưng được mã hóa và chuẩn hóa đúng cách để 
# cải thiện hiệu suất của mô hình.
def prepare_abalone_feature_groups() -> dict[str, list[str] | str]:
    """Trả về các nhóm đặc trưng dùng trong notebook tiền xử lý."""
    categorical_cols = ["Sex"]
    size_cols = ["Length", "Diameter", "Height"]
    weight_cols = ["WholeWeight", "ShuckedWeight", "VisceraWeight", "ShellWeight"]
    numeric_cols = size_cols + weight_cols

    return {
        "categorical_cols": categorical_cols,
        "size_cols": size_cols,
        "weight_cols": weight_cols,
        "numeric_cols": numeric_cols,
        "target_col": "Rings",
    }


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Điểm mở rộng cho feature engineering bổ sung."""
    return df.copy()

# Hàm make_onehot_encoder tạo một OneHotEncoder tương thích với nhiều phiên bản scikit-learn,
#  đảm bảo rằng nó có thể được sử dụng trong các môi trường khác nhau mà không gặp vấn đề về tham số.
#  Điều này giúp mã nguồn trở nên linh hoạt và dễ bảo trì hơn khi làm việc với các phiên bản 
# scikit-learn khác nhau.
def make_onehot_encoder() -> OneHotEncoder:
    """Tạo OneHotEncoder tương thích nhiều phiên bản scikit-learn."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# Hàm transform_with_preprocessor giúp áp dụng một ColumnTransformer 
# đã được fit trên tập huấn luyện để biến đổi cả tập huấn luyện và tập kiểm tra,
#  đồng thời trả về kết quả dưới dạng DataFrame với tên cột rõ ràng. 
# Điều này rất hữu ích để đảm bảo rằng các bước tiền xử lý được áp dụng nhất quán cho cả hai 
# tập dữ liệu và giúp dễ dàng theo dõi các đặc trưng sau khi biến đổi.
def transform_with_preprocessor(
    preprocessor: ColumnTransformer,
    x_train_df: pd.DataFrame,
    x_test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit trên train, transform cho cả train/test va trả về DataFrame."""
    x_train_transformed = preprocessor.fit_transform(x_train_df)
    x_test_transformed = preprocessor.transform(x_test_df)
    feature_names = preprocessor.get_feature_names_out()

    x_train_result = pd.DataFrame(
        x_train_transformed,
        columns=feature_names,
        index=x_train_df.index,
    )
    x_test_result = pd.DataFrame(
        x_test_transformed,
        columns=feature_names,
        index=x_test_df.index,
    )
    return x_train_result, x_test_result

# Hàm apply_log1p_to_columns áp dụng biến đổi log1p cho một nhóm cột số để giảm độ lệch của dữ liệu.
#  Điều này rất hữu ích khi dữ liệu có phân phối lệch và giúp cải thiện hiệu suất của các mô hình học máy bằng cách làm cho dữ liệu trở nên gần với phân phối chuẩn hơn.
def apply_log1p_to_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Áp dụng log1p cho một nhóm cột số để giảm độ lệch."""
    transformed_df = df.copy()
    transformed_df[columns] = np.log1p(transformed_df[columns])
    return transformed_df

# Hàm build_encoded_preprocessor tạo một ColumnTransformer chỉ mã hóa các biến phân loại
#  bằng OneHotEncoder, trong khi giữ nguyên các cột còn lại. 
# Điều này rất hữu ích khi bạn muốn chỉ mã hóa một số cột nhất định mà không ảnh hưởng đến 
# các cột khác trong quá trình tiền xử lý.
def build_encoded_preprocessor(categorical_cols: list[str]) -> ColumnTransformer:
    """Tạo bộ xử lý chỉ mã hóa biến phân loại."""
    return ColumnTransformer(
        transformers=[
            ("sex", make_onehot_encoder(), categorical_cols),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

# Hàm build_standard_scaled_preprocessor tạo một ColumnTransformer kết hợp 
# OneHotEncoder cho các biến phân loại và StandardScaler cho các biến số, 
# giúp chuẩn hóa dữ liệu số và mã hóa dữ liệu phân loại trong cùng một bước tiền xử lý.
def build_standard_scaled_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> ColumnTransformer:
    """Tạo bộ xử lý standard scaling cho biến số."""
    return ColumnTransformer(
        transformers=[
            ("sex", make_onehot_encoder(), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ], # áp dụng OneHotEncoder cho các cột phân loại và StandardScaler cho các cột số
        remainder="drop", # loại bỏ các cột không được chỉ định trong transformers
        verbose_feature_names_out=False,
    )

# Hàm build_robust_scaled_preprocessor tạo một ColumnTransformer 
# kết hợp OneHotEncoder cho các biến phân loại và RobustScaler cho các biến số,
#  giúp chuẩn hóa dữ liệu số một cách hiệu quả ngay cả khi có outliers trong dữ liệu.
def build_robust_scaled_preprocessor(
    categorical_cols: list[str],
    numeric_cols: list[str], 
) -> ColumnTransformer:
    """Tạo bộ xử lý robust scaling cho biến số."""
    return ColumnTransformer(
        transformers=[
            ("sex", make_onehot_encoder(), categorical_cols),
            ("num", RobustScaler(), numeric_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
