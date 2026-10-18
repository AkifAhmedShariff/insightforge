import pandas as pd
import pytest

from data_analysis import get_category_share


def test_category_share():
    df = pd.DataFrame({"Region": ["North", "South"], "Sales": [25, 75]})

    assert get_category_share(df, "Region", "Sales") == {"North": 25.0, "South": 75.0}


def test_category_share_rejects_zero_total():
    df = pd.DataFrame({"Region": ["North"], "Sales": [0]})

    with pytest.raises(ValueError, match="must not be 0"):
        get_category_share(df, "Region", "Sales")


def test_category_share_missing_column():
    df = pd.DataFrame({"Sales": [10]})

    with pytest.raises(KeyError, match="Column not found"):
        get_category_share(df, "Region", "Sales")


def test_category_share_adds_rows_in_category():
    df = pd.DataFrame({"Region": ["North", "North", "South"], "Sales": [10, 10, 20]})

    assert get_category_share(df, "Region", "Sales") == {"North": 50.0, "South": 50.0}
