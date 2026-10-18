import pandas as pd
import pytest

from data_analysis import get_value_range


def test_value_range():
    df = pd.DataFrame({"Sales": [10, 50, 30]})

    assert get_value_range(df, "Sales") == 40.0


def test_value_range_rejects_empty_column():
    df = pd.DataFrame({"Sales": [None, None]})

    with pytest.raises(ValueError, match="no numeric values"):
        get_value_range(df, "Sales")
