import pandas as pd
import pytest

from data_analysis import get_daily_average


def test_daily_average():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"], "Sales": [10, 20]})

    assert get_daily_average(df, "Date", "Sales") == 15.0


def test_daily_average_ignores_invalid_dates():
    df = pd.DataFrame({"Date": ["2024-01-01", None], "Sales": [10, 99]})

    assert get_daily_average(df, "Date", "Sales") == 10.0


def test_daily_average_no_valid_dates():
    df = pd.DataFrame({"Date": [None], "Sales": [10]})

    with pytest.raises(ValueError, match="no valid dates"):
        get_daily_average(df, "Date", "Sales")


def test_daily_average_missing_date_column():
    df = pd.DataFrame({"Sales": [10]})

    with pytest.raises(KeyError, match="Column not found"):
        get_daily_average(df, "Date", "Sales")
