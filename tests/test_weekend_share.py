import pandas as pd
import pytest

from data_analysis import get_weekend_share


def test_weekend_share():
    df = pd.DataFrame({"Date": ["2024-01-07", "2024-01-08"], "Sales": [25, 75]})

    assert get_weekend_share(df, "Date", "Sales") == 25.0


def test_weekend_share_ignores_invalid_dates():
    df = pd.DataFrame({"Date": ["2024-01-07", "not a date", "2024-01-08"], "Sales": [10, 100, 30]})

    assert get_weekend_share(df, "Date", "Sales") == 25.0


def test_weekend_share_rejects_zero_total():
    df = pd.DataFrame({"Date": ["2024-01-07"], "Sales": [0]})

    with pytest.raises(ValueError, match="must not be 0"):
        get_weekend_share(df, "Date", "Sales")


def test_weekend_share_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_weekend_share(df, "Date", "Sales")


def test_weekend_share_includes_saturday():
    df = pd.DataFrame({"Date": ["2024-01-06", "2024-01-08"], "Sales": [50, 50]})

    assert get_weekend_share(df, "Date", "Sales") == 50.0
