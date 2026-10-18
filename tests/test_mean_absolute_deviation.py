import pandas as pd
import pytest

from data_analysis import get_mean_absolute_deviation


def test_mean_absolute_deviation():
    df = pd.DataFrame({"Sales": [1, 2, 3]})

    assert get_mean_absolute_deviation(df, "Sales") == 0.67


def test_mean_absolute_deviation_skips_missing_values():
    df = pd.DataFrame({"Sales": [4, None, 4]})

    assert get_mean_absolute_deviation(df, "Sales") == 0.0


def test_mean_absolute_deviation_rejects_empty_column():
    df = pd.DataFrame({"Sales": [float("nan")]})

    with pytest.raises(ValueError, match="no values"):
        get_mean_absolute_deviation(df, "Sales")


def test_mean_absolute_deviation_uses_mean_for_skewed_values():
    df = pd.DataFrame({"Sales": [1, 2, 6]})

    assert get_mean_absolute_deviation(df, "Sales") == 2.0
