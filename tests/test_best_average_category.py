import pandas as pd
import pytest

from data_analysis import get_best_average_category


def test_best_average_category():
    df = pd.DataFrame({"Region": ["North", "South"], "Sales": [10, 20]})

    assert get_best_average_category(df, "Region", "Sales") == "South"


def test_best_average_category_rejects_empty_column():
    df = pd.DataFrame({"Region": ["North"], "Sales": [float("nan")]})

    with pytest.raises(ValueError, match="no values"):
        get_best_average_category(df, "Region", "Sales")


def test_best_average_category_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_best_average_category(df, "Region", "Sales")
