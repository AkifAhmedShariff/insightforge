import pandas as pd

from data_analysis import fill_missing_with_median


def test_fill_missing_with_median():
    df = pd.DataFrame({"Sales": [10, None, 30]})

    result = fill_missing_with_median(df, "Sales")

    assert list(result["Sales"]) == [10.0, 20.0, 30.0]
