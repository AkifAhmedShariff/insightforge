import pandas as pd
import pytest

from data_analysis import get_max_drop


def test_max_drop():
    df = pd.DataFrame({"Price": [10, 4, 8]})

    assert get_max_drop(df, "Price") == 6.0


def test_max_drop_no_fall_is_zero():
    df = pd.DataFrame({"Price": [1, 2, None, 5]})

    assert get_max_drop(df, "Price") == 0.0


def test_max_drop_empty_column():
    df = pd.DataFrame({"Price": [None, None]})

    with pytest.raises(ValueError, match="no values"):
        get_max_drop(df, "Price")


def test_max_drop_missing_column():
    df = pd.DataFrame({"Price": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_max_drop(df, "Cost")
