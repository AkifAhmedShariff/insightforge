import pandas as pd
import pytest

from data_analysis import get_latest_value


def test_latest_value_single_row():
    df = pd.DataFrame({"Date": ["2024-05-01"], "Sales": [300]})

    assert get_latest_value(df, "Date", "Sales") == 300


def test_latest_value_rejects_invalid_dates():
    df = pd.DataFrame({"Date": ["not a date"], "Sales": [300]})

    with pytest.raises(ValueError, match="no valid dates"):
        get_latest_value(df, "Date", "Sales")


def test_latest_value_picks_most_recent_date():
    df = pd.DataFrame({
        "Date": ["2024-05-01", "2024-07-01", "2024-06-01"],
        "Sales": [100, 300, 200],
    })

    assert get_latest_value(df, "Date", "Sales") == 300
