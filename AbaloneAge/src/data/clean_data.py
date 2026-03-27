"""Cac ham tien xu ly va lam sach du lieu."""

import pandas as pd


def normalize_column_names(columns) -> list[str]:
    """Chuẩn hoá tên cột theo định dạng lowercase, bỏ khoảng trắng dư."""
    return [str(column).strip().lower() for column in columns]

# hàm làm sạch dữ liệu Abalone, bao gồm chuẩn hoá tên cột, loại bỏ khoảng trắng dư và tuỳ chọn loại bỏ các dòng trùng lặp. Hàm này giúp đảm bảo rằng dữ liệu đầu vào đã được làm sạch và sẵn sàng cho các bước xử lý tiếp theo trong pipeline.
def clean_abalone_data(
    df: pd.DataFrame,
    *,
    drop_duplicates: bool = False,
    strip_categorical_values: bool = True,
) -> pd.DataFrame:
    """Trả về dữ liệu sau khi đã được làm sạch."""
    cleaned_df = df.copy()
    cleaned_df.columns = normalize_column_names(cleaned_df.columns)

    if strip_categorical_values:
        object_cols = cleaned_df.select_dtypes(include=["object", "string"]).columns
        for column in object_cols:
            cleaned_df[column] = cleaned_df[column].astype("string").str.strip()

    # Tuỳ chọn loại bỏ các dòng trùng lặp để đảm bảo rằng dữ liệu không chứa các bản ghi trùng lặp, giúp cải thiện chất lượng dữ liệu và độ chính xác của mô hình học máy.
    if drop_duplicates:
        cleaned_df = cleaned_df.drop_duplicates().copy()

    return cleaned_df


def summarize_data_quality(df: pd.DataFrame) -> dict:
    """Tổng hợp nhanh missing values và số dòng trùng lặp."""
    missing_count = int(df.isna().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_count": missing_count,
        "duplicate_count": duplicate_count,
    }
