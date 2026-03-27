import pandas as pd

from src.data.clean_data import clean_abalone_data, summarize_data_quality
from src.data.split_data import attach_target, split_features_target

# Các hàm trong module này tập trung vào việc kiểm tra các chức năng làm sạch dữ liệu, tách features và target, cũng như đảm bảo rằng các bước này hoạt động đúng cách để chuẩn bị dữ liệu cho mô hình học máy. 
# Chúng giúp đảm bảo rằng dữ liệu đã được làm sạch và tách ra một cách chính xác, 
# từ đó cải thiện chất lượng của mô hình và độ chính xác của dự đoán


def test_clean_abalone_data_normalizes_column_names():
    df = pd.DataFrame({" Length ": [1.0], " Rings ": [10]})
    cleaned_df = clean_abalone_data(df)

    assert cleaned_df.columns.tolist() == ["length", "rings"]


def test_summarize_data_quality_reports_missing_and_duplicates():
    df = pd.DataFrame({"a": [1, 1], "b": [None, None]})

    summary = summarize_data_quality(df)

    assert summary["row_count"] == 2
    assert summary["missing_count"] == 2
    assert summary["duplicate_count"] == 1


def test_split_features_target_and_attach_target_keep_alignment():
    df = pd.DataFrame(
        {
            "Sex": ["M", "F", "I", "M"],
            "Length": [0.4, 0.5, 0.3, 0.6],
            "Rings": [10, 11, 8, 12],
        }
    )

    x_train, x_test, y_train, y_test = split_features_target(df, "Rings", test_size=0.5, random_state=42)
    rebuilt_train = attach_target(x_train, y_train, "Rings")
    rebuilt_test = attach_target(x_test, y_test, "Rings")

    assert "Rings" not in x_train.columns
    assert "Rings" in rebuilt_train.columns
    assert len(rebuilt_train) + len(rebuilt_test) == len(df)
