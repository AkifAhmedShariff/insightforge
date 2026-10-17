import pandas as pd
import pytest

from data_analysis import get_average_by_category


def test_average_by_category():
    df = pd.DataFrame({
        "Region": ["North", "South"],
        "Sales": [100, 250],
    })

    assert get_average_by_category(df, "Region", "Sales") == {
        "North": 100.0,
        "South": 250.0,
    }


def test_average_by_category_missing_column():
    df = pd.DataFrame({"Sales": [10, 20]})

    with pytest.raises(KeyError, match="Column not found"):
        get_average_by_category(df, "Region", "Sales")


def test_average_by_category_with_multiple_rows():
    df = pd.DataFrame({
        "Region": ["North", "North", "South"],
        "Sales": [100, 200, 250],
    })

    assert get_average_by_category(df, "Region", "Sales") == {
        "North": 150.0,
        "South": 250.0,
    }
