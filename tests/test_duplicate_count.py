import pandas as pd
import pytest

from data_analysis import count_duplicate_rows


def test_count_duplicate_rows_none():
    df = pd.DataFrame({"Region": ["North", "South"], "Sales": [1, 2]})

    assert count_duplicate_rows(df) == 0


def test_count_duplicate_rows_subset():
    df = pd.DataFrame({"Region": ["North", "South"], "Sales": [1, 1]})

    assert count_duplicate_rows(df, ["Region"]) == 0


def test_count_duplicate_rows_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        count_duplicate_rows(df, ["Region"])
