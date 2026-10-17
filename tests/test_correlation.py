import pandas as pd
import pytest

from data_analysis import get_correlation


def test_correlation_of_increasing_columns_is_positive():
    df = pd.DataFrame({
        "Ads": [1, 2, 3, 4],
        "Sales": [10, 20, 30, 40],
    })

    assert get_correlation(df, "Ads", "Sales") == 1.0


def test_correlation_needs_two_rows():
    df = pd.DataFrame({"Ads": [1, None], "Sales": [10, 20]})

    with pytest.raises(ValueError, match="at least two complete rows"):
        get_correlation(df, "Ads", "Sales")


def test_correlation_is_pearson_not_rank_based():
    df = pd.DataFrame({
        "Ads": [1, 2, 3, 4],
        "Sales": [1, 2, 3, 100],
    })

    assert get_correlation(df, "Ads", "Sales") == 0.785
