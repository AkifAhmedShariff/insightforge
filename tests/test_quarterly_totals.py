import pandas as pd
import pytest

from data_analysis import get_quarterly_totals


def test_quarterly_totals():
    df = pd.DataFrame({
        "Date": ["2024-01-15", "2024-05-01", "2024-05-20"],
        "Sales": [100, 50, 25],
    })

    assert get_quarterly_totals(df, "Date", "Sales") == {"2024-Q1": 100.0, "2024-Q2": 75.0}


def test_quarterly_totals_ignores_invalid_dates():
    df = pd.DataFrame({
        "Date": ["2024-11-01", "not a date"],
        "Sales": [20, 10],
    })

    assert get_quarterly_totals(df, "Date", "Sales") == {"2024-Q4": 20.0}


def test_quarterly_totals_missing_date_column():
    df = pd.DataFrame({"Sales": [10]})

    with pytest.raises(KeyError, match="Column not found"):
        get_quarterly_totals(df, "Date", "Sales")


def test_quarterly_totals_last_month_of_quarter():
    df = pd.DataFrame({
        "Date": ["2024-03-31", "2024-12-01"],
        "Sales": [10, 30],
    })

    assert get_quarterly_totals(df, "Date", "Sales") == {"2024-Q1": 10.0, "2024-Q4": 30.0}
