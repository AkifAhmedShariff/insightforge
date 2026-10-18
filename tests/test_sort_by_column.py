import pandas as pd
import pytest

from data_analysis import sort_by_column


def test_sort_by_column_descending():
    df = pd.DataFrame({"Product": ["A", "B", "C"], "Sales": [100, 300, 200]})

    assert list(sort_by_column(df, "Sales")["Product"]) == ["B", "C", "A"]


def test_sort_by_column_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        sort_by_column(df, "Region")
