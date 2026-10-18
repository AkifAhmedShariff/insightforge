import pandas as pd
import pytest

from data_analysis import get_geometric_mean


def test_geometric_mean():
    df = pd.DataFrame({"Growth": [2, 8, None]})

    assert get_geometric_mean(df, "Growth") == 4.0


def test_geometric_mean_negative_value():
    df = pd.DataFrame({"Growth": [2, -8]})

    with pytest.raises(ValueError, match="greater than 0"):
        get_geometric_mean(df, "Growth")


def test_geometric_mean_empty_column():
    df = pd.DataFrame({"Growth": [None]})

    with pytest.raises(ValueError, match="no values"):
        get_geometric_mean(df, "Growth")


def test_geometric_mean_missing_column():
    df = pd.DataFrame({"Growth": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_geometric_mean(df, "Sales")


def test_geometric_mean_zero_value():
    df = pd.DataFrame({"Growth": [0, 4]})

    with pytest.raises(ValueError, match="greater than 0"):
        get_geometric_mean(df, "Growth")
