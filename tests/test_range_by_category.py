import pandas as pd
import pytest

from data_analysis import get_range_by_category


def test_range_by_category():
    df = pd.DataFrame({"Region": ["North", "North", "South", "South"], "Sales": [1, 4, 2, 2]})

    assert get_range_by_category(df, "Region", "Sales") == {"North": 3.0, "South": 0.0}


def test_range_by_category_single_row():
    df = pd.DataFrame({"Region": ["North"], "Sales": [7]})

    assert get_range_by_category(df, "Region", "Sales") == {"North": 0.0}


def test_range_by_category_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_range_by_category(df, "Region", "Sales")


def test_range_by_category_unsorted_values():
    df = pd.DataFrame({"Region": ["North", "North", "North"], "Sales": [20, 10, 15]})

    assert get_range_by_category(df, "Region", "Sales") == {"North": 10.0}
