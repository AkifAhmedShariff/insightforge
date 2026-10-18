import pandas as pd
import pytest

from data_analysis import get_percent_of_max


def test_percent_of_max():
    df = pd.DataFrame({"Sales": [5, 5]})

    assert get_percent_of_max(df, "Sales") == [100.0, 100.0]


def test_percent_of_max_missing_value_is_none():
    df = pd.DataFrame({"Sales": [None, 4]})

    assert get_percent_of_max(df, "Sales") == [None, 100.0]


def test_percent_of_max_rejects_zero_maximum():
    df = pd.DataFrame({"Sales": [0, 0]})

    with pytest.raises(ValueError, match="must not be 0"):
        get_percent_of_max(df, "Sales")
