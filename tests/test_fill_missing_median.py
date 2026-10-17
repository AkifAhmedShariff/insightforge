import pandas as pd

from data_analysis import fill_missing_with_median


def test_fill_missing_with_median():
    df = pd.DataFrame({"Sales": [10, None, 30]})

    result = fill_missing_with_median(df, "Sales")

    assert list(result["Sales"]) == [10.0, 20.0, 30.0]


def test_fill_missing_with_median_does_not_modify_input():
    df = pd.DataFrame({"Sales": [10, None, 30]})

    fill_missing_with_median(df, "Sales")

    assert df["Sales"].isna().sum() == 1
