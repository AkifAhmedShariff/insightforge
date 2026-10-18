import pandas as pd
import pytest

from data_analysis import get_duplicate_values


def test_duplicate_values():
    df = pd.DataFrame({"Code": [3, 1, 3, 2, 1]})

    assert get_duplicate_values(df, "Code") == [1, 3]


def test_duplicate_values_ignores_missing_values():
    df = pd.DataFrame({"Code": [1, 2, None, None]})

    assert get_duplicate_values(df, "Code") == []


def test_duplicate_values_missing_column():
    df = pd.DataFrame({"Code": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_duplicate_values(df, "Id")


def test_duplicate_values_listed_once():
    df = pd.DataFrame({"Code": ["a", "a", "a", "b"]})

    assert get_duplicate_values(df, "Code") == ["a"]
