import pandas as pd
import pytest

from data_analysis import get_recent_average


def test_recent_average():
    df = pd.DataFrame({"Sales": [2, 4, 6]})

    assert get_recent_average(df, "Sales") == 4.0


def test_recent_average_skips_missing_values():
    df = pd.DataFrame({"Sales": [1, None, 3]})

    assert get_recent_average(df, "Sales") == 2.0


def test_recent_average_rejects_small_n():
    df = pd.DataFrame({"Sales": [1, 2]})

    with pytest.raises(ValueError, match="at least 1"):
        get_recent_average(df, "Sales", n=0)
