import pandas as pd
import pytest

from data_analysis import get_monthly_counts


def test_monthly_counts():
    df = pd.DataFrame({"Date": ["2024-10-01", "2024-10-15", "2024-11-02"]})

    assert get_monthly_counts(df, "Date") == {"2024-10": 2, "2024-11": 1}


def test_monthly_counts_ignores_invalid_dates():
    df = pd.DataFrame({"Date": ["2024-12-01", "not a date"]})

    assert get_monthly_counts(df, "Date") == {"2024-12": 1}


def test_monthly_counts_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_monthly_counts(df, "Date")
