import pandas as pd
import pytest

from data_analysis import get_median_by_category


def test_median_by_category():
    df = pd.DataFrame({
        "Region": ["North", "North", "South"],
        "Sales": [100, 200, 50],
    })

    assert get_median_by_category(df, "Region", "Sales") == {
        "North": 150.0,
        "South": 50.0,
    }


def test_median_by_category_missing_column():
    df = pd.DataFrame({"Sales": [10]})

    with pytest.raises(KeyError, match="Column not found"):
        get_median_by_category(df, "Region", "Sales")


def test_median_by_category_differs_from_mean():
    df = pd.DataFrame({
        "Region": ["North", "North", "North"],
        "Sales": [100, 200, 900],
    })

    assert get_median_by_category(df, "Region", "Sales") == {"North": 200.0}
