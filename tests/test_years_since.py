import pandas as pd
import pytest

from data_analysis import get_years_since


def test_years_since():
    df = pd.DataFrame({"Joined": ["2020-06-01", "2023-01-01"]})

    assert get_years_since(df, "Joined", "2024-06-01") == [4, 1]


def test_years_since_invalid_date_is_none():
    df = pd.DataFrame({"Joined": ["2022-03-01", "not a date"]})

    assert get_years_since(df, "Joined", "2024-03-01") == [2, None]


def test_years_since_missing_column():
    df = pd.DataFrame({"Joined": ["2024-01-01"]})

    with pytest.raises(KeyError, match="Column not found"):
        get_years_since(df, "Date", "2024-01-01")
