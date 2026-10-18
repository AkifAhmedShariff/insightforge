import pandas as pd
import pytest

from data_analysis import get_cumulative_percentage


def test_cumulative_percentage():
    df = pd.DataFrame({"Sales": [40]})

    assert get_cumulative_percentage(df, "Sales") == [100.0]


def test_cumulative_percentage_missing_value_counts_as_zero():
    df = pd.DataFrame({"Sales": [None, 5]})

    assert get_cumulative_percentage(df, "Sales") == [0.0, 100.0]


def test_cumulative_percentage_rejects_zero_total():
    df = pd.DataFrame({"Sales": [0, 0]})

    with pytest.raises(ValueError, match="must not be 0"):
        get_cumulative_percentage(df, "Sales")
