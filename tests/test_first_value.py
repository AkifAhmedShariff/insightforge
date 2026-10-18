import pandas as pd
import pytest

from data_analysis import get_first_value


def test_first_value():
    df = pd.DataFrame({"Sales": [5, 3, 8]})

    assert get_first_value(df, "Sales") == 5


def test_first_value_text_column():
    df = pd.DataFrame({"Region": ["North", "South"]})

    assert get_first_value(df, "Region") == "North"


def test_first_value_rejects_empty_column():
    df = pd.DataFrame({"Sales": [float("nan"), float("nan")]})

    with pytest.raises(ValueError, match="no values"):
        get_first_value(df, "Sales")


def test_first_value_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_first_value(df, "Profit")
