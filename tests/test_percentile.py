import pandas as pd
import pytest

from data_analysis import get_percentile


def test_percentile_returns_median_for_50():
    df = pd.DataFrame({"Sales": [10, 20, 30]})

    assert get_percentile(df, "Sales", 50) == 20.0


def test_percentile_rejects_values_above_100():
    df = pd.DataFrame({"Sales": [10, 20, 30]})

    with pytest.raises(ValueError, match="between 0 and 100"):
        get_percentile(df, "Sales", 150)


def test_percentile_accepts_0_and_100():
    df = pd.DataFrame({"Sales": [10, 20, 30]})

    assert get_percentile(df, "Sales", 0) == 10.0
    assert get_percentile(df, "Sales", 100) == 30.0
