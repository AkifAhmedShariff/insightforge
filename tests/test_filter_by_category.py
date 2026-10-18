import pandas as pd
import pytest

from data_analysis import filter_by_category


def test_filter_by_category():
    df = pd.DataFrame({"Region": ["North", "South", "East"], "Sales": [1, 2, 3]})

    result = filter_by_category(df, "Region", ["North", "East"])

    assert list(result["Sales"]) == [1, 3]


def test_filter_by_category_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        filter_by_category(df, "Region", ["North"])
