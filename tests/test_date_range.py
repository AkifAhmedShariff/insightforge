import pandas as pd
import pytest

from data_analysis import get_date_range


def test_date_range():
    df = pd.DataFrame({"Date": ["2024-03-15"]})

    assert get_date_range(df, "Date") == {"start": "2024-03-15", "end": "2024-03-15"}


def test_date_range_rejects_no_valid_dates():
    df = pd.DataFrame({"Date": ["not a date"]})

    with pytest.raises(ValueError, match="no valid dates"):
        get_date_range(df, "Date")


def test_date_range_missing_column():
    df = pd.DataFrame({"Sales": [10]})

    with pytest.raises(KeyError, match="Column not found"):
        get_date_range(df, "Date")
