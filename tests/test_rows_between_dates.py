import pandas as pd
import pytest

from data_analysis import filter_rows_between_dates


def test_rows_between_dates():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-15", "2024-02-01"], "Sales": [1, 2, 3]})

    result = filter_rows_between_dates(df, "Date", "2024-01-01", "2024-01-31")

    assert result["Sales"].tolist() == [1, 2]


def test_rows_between_dates_excludes_invalid_dates():
    df = pd.DataFrame({"Date": ["2024-01-10", "not a date"], "Sales": [1, 2]})

    result = filter_rows_between_dates(df, "Date", "2024-01-01", "2024-01-31")

    assert result["Sales"].tolist() == [1]


def test_rows_between_dates_rejects_reversed_range():
    df = pd.DataFrame({"Date": ["2024-01-10"]})

    with pytest.raises(ValueError, match="start must not be after end"):
        filter_rows_between_dates(df, "Date", "2024-02-01", "2024-01-01")


def test_rows_between_dates_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        filter_rows_between_dates(df, "Date", "2024-01-01", "2024-01-31")


def test_rows_between_dates_includes_end_date():
    df = pd.DataFrame({"Date": ["2024-01-01", "2024-01-15", "2024-01-20"], "Sales": [1, 2, 3]})

    result = filter_rows_between_dates(df, "Date", "2024-01-01", "2024-01-15")

    assert result["Sales"].tolist() == [1, 2]
