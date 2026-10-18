import pandas as pd
import pytest

from data_analysis import count_above_by_category


def test_count_above_by_category():
    df = pd.DataFrame({"Region": ["North", "North", "South"], "Sales": [5, 20, 30]})

    assert count_above_by_category(df, "Region", "Sales", 10) == {"North": 1, "South": 1}


def test_count_above_by_category_skips_missing_values():
    df = pd.DataFrame({"Region": ["North", "South", "South"], "Sales": [20, None, 15]})

    assert count_above_by_category(df, "Region", "Sales", 10) == {"North": 1, "South": 1}


def test_count_above_by_category_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        count_above_by_category(df, "Region", "Sales", 0)
