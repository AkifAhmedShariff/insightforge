import pandas as pd
import pytest

from data_analysis import get_quartiles


def test_quartiles_of_constant_column():
    df = pd.DataFrame({"Sales": [5, 5, 5, 5]})

    assert get_quartiles(df, "Sales") == {"q1": 5.0, "q2": 5.0, "q3": 5.0}


def test_quartiles_reject_empty_column():
    df = pd.DataFrame({"Sales": [None]})

    with pytest.raises(ValueError, match="no numeric values"):
        get_quartiles(df, "Sales")
