import pandas as pd
import pytest

from data_analysis import get_column_differences


def test_column_differences():
    df = pd.DataFrame({"Revenue": [10, 7.5], "Cost": [4, 2.25]})

    assert get_column_differences(df, "Revenue", "Cost") == [6.0, 5.25]


def test_column_differences_missing_value_is_none():
    df = pd.DataFrame({"Revenue": [10, None], "Cost": [3, 1]})

    assert get_column_differences(df, "Revenue", "Cost") == [7.0, None]


def test_column_differences_missing_column():
    df = pd.DataFrame({"Revenue": [10]})

    with pytest.raises(KeyError, match="Column not found"):
        get_column_differences(df, "Revenue", "Cost")
