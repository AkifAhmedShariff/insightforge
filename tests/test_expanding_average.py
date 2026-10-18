import pandas as pd
import pytest

from data_analysis import get_expanding_average


def test_expanding_average():
    df = pd.DataFrame({"Sales": [2, 4, 6]})

    assert get_expanding_average(df, "Sales") == [2.0, 3.0, 4.0]


def test_expanding_average_missing_value_is_none():
    df = pd.DataFrame({"Sales": [2, 4, None]})

    assert get_expanding_average(df, "Sales") == [2.0, 3.0, None]


def test_expanding_average_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_expanding_average(df, "Profit")
