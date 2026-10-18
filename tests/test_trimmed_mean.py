import pandas as pd
import pytest

from data_analysis import get_trimmed_mean


def test_trimmed_mean_small_column_keeps_all_values():
    df = pd.DataFrame({"Sales": [3, 1, 2]})

    assert get_trimmed_mean(df, "Sales") == 2.0


def test_trimmed_mean_zero_proportion():
    df = pd.DataFrame({"Sales": [1, 2, 6]})

    assert get_trimmed_mean(df, "Sales", proportion=0) == 3.0


def test_trimmed_mean_rejects_large_proportion():
    df = pd.DataFrame({"Sales": [1, 2]})

    with pytest.raises(ValueError, match="proportion must be"):
        get_trimmed_mean(df, "Sales", proportion=0.5)


def test_trimmed_mean_drops_both_ends():
    df = pd.DataFrame({"Sales": [1, 2, 3, 4, 100]})

    assert get_trimmed_mean(df, "Sales", proportion=0.2) == 3.0
