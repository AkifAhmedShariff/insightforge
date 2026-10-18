import pandas as pd
import pytest

from data_analysis import get_rows_above_mean


def test_rows_above_mean():
    df = pd.DataFrame({"Sales": [1, 5]})

    assert get_rows_above_mean(df, "Sales")["Sales"].tolist() == [5]


def test_rows_above_mean_skips_missing_values():
    df = pd.DataFrame({"Sales": [None, 2, 10]})

    assert get_rows_above_mean(df, "Sales")["Sales"].tolist() == [10.0]


def test_rows_above_mean_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_rows_above_mean(df, "Profit")
