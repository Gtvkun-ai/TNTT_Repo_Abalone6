from src.data.clean_data import clean_abalone_data


def test_clean_abalone_data_normalizes_column_names():
    import pandas as pd

    df = pd.DataFrame({" Length ": [1.0], " Rings ": [10]})
    cleaned_df = clean_abalone_data(df)

    assert cleaned_df.columns.tolist() == ["length", "rings"]
