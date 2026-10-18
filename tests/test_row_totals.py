import pandas as pd
import pytest

from data_analysis import get_row_totals


def test_row_totals():
    df = pd.DataFrame({"Online": [1, 2], "Store": [0, 0]})

    assert get_row_totals(df, ["Online", "Store"]) == [1.0, 2.0]


def test_row_totals_missing_value_counts_as_zero():
    df = pd.DataFrame({"Online": [None, 1], "Store": [5, 0]})

    assert get_row_totals(df, ["Online", "Store"]) == [5.0, 1.0]


def test_row_totals_missing_column():
    df = pd.DataFrame({"Online": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_row_totals(df, ["Online", "Store"])
