import pandas as pd
import pytest

from data_analysis import get_monthly_totals


def test_monthly_totals():
    df = pd.DataFrame({
        "Date": ["2024-01-01", "2024-02-02", "2024-02-02"],
        "Sales": [100, 50, 25],
    })

    assert get_monthly_totals(df, "Date", "Sales") == {
        "2024-01": 100.0,
        "2024-02": 75.0,
    }


def test_monthly_totals_missing_date_column():
    df = pd.DataFrame({"Sales": [10]})

    with pytest.raises(KeyError, match="Column not found"):
        get_monthly_totals(df, "Date", "Sales")


def test_monthly_totals_groups_by_month_not_day():
    df = pd.DataFrame({
        "Date": ["2024-01-15", "2024-01-20"],
        "Sales": [100, 50],
    })

    assert get_monthly_totals(df, "Date", "Sales") == {"2024-01": 150.0}
