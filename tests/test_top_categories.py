import pandas as pd
import pytest

from data_analysis import get_top_categories


def test_top_categories():
    df = pd.DataFrame({"Region": ["North", "North"], "Sales": [10, 20]})

    assert get_top_categories(df, "Region", "Sales") == ["North"]


def test_top_categories_rejects_small_n():
    df = pd.DataFrame({"Region": ["North"], "Sales": [10]})

    with pytest.raises(ValueError, match="at least 1"):
        get_top_categories(df, "Region", "Sales", n=0)


def test_top_categories_missing_column():
    df = pd.DataFrame({"Sales": [10]})

    with pytest.raises(KeyError, match="Column not found"):
        get_top_categories(df, "Region", "Sales")
