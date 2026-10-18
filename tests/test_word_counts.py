import pandas as pd
import pytest

from data_analysis import get_word_counts


def test_word_counts():
    df = pd.DataFrame({"Notes": ["hello world", "one"]})

    assert get_word_counts(df, "Notes") == [2, 1]


def test_word_counts_missing_value_is_none():
    df = pd.DataFrame({"Notes": ["a b c", None]})

    assert get_word_counts(df, "Notes") == [3, None]


def test_word_counts_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_word_counts(df, "Notes")
