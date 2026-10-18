import pandas as pd
import pytest

from data_analysis import get_days_since_first


def test_days_since_first():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-03", "2024-01-10"]})

    assert get_days_since_first(df, "Date") == [0, 2, 9]


def test_days_since_first_invalid_date_is_none():
    df = pd.DataFrame({"Date": ["2024-01-01", "not a date", "2024-01-02"]})

    assert get_days_since_first(df, "Date") == [0, None, 1]


def test_days_since_first_rejects_no_valid_dates():
    df = pd.DataFrame({"Date": ["not a date"]})

    with pytest.raises(ValueError, match="no valid dates"):
        get_days_since_first(df, "Date")


def test_days_since_first_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_days_since_first(df, "Date")


def test_days_since_first_unsorted_dates():
    df = pd.DataFrame({"Date": ["2024-01-05", "2024-01-01"]})

    assert get_days_since_first(df, "Date") == [4, 0]
