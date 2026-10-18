import pandas as pd
import pytest

from data_analysis import get_days_between


def test_days_between_same_day_is_zero():
    df = pd.DataFrame({"Ordered": ["2024-03-01"], "Shipped": ["2024-03-01"]})

    assert get_days_between(df, "Ordered", "Shipped") == [0]


def test_days_between_missing_column():
    df = pd.DataFrame({"Ordered": ["2024-03-01"]})

    with pytest.raises(KeyError, match="Column not found"):
        get_days_between(df, "Ordered", "Shipped")


def test_days_between_is_end_minus_start():
    df = pd.DataFrame({"Ordered": ["2024-03-01"], "Shipped": ["2024-03-05"]})

    assert get_days_between(df, "Ordered", "Shipped") == [4]
