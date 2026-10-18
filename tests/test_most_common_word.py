import pandas as pd
import pytest

from data_analysis import get_most_common_word


def test_most_common_word():
    df = pd.DataFrame({"Notes": ["red apple", "green apple", None]})

    assert get_most_common_word(df, "Notes") == "apple"


def test_most_common_word_empty_column():
    df = pd.DataFrame({"Notes": [None, "  "]})

    with pytest.raises(ValueError, match="no words"):
        get_most_common_word(df, "Notes")


def test_most_common_word_missing_column():
    df = pd.DataFrame({"Notes": ["a"]})

    with pytest.raises(KeyError, match="Column not found"):
        get_most_common_word(df, "Text")


def test_most_common_word_ignores_case():
    df = pd.DataFrame({"Notes": ["Red apple", "red car", "RED box", "apple"]})

    assert get_most_common_word(df, "Notes") == "red"
