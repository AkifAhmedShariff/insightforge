import pandas as pd
import pytest

from data_analysis import count_rows_starting_with


def test_count_rows_starting_with():
    df = pd.DataFrame({"Item": ["Apple pie", "banana", "apple tart", None]})

    assert count_rows_starting_with(df, "Item", "apple") == 2


def test_count_rows_starting_with_no_matches():
    df = pd.DataFrame({"Item": ["kiwi"]})

    assert count_rows_starting_with(df, "Item", "pear") == 0


def test_count_rows_starting_with_missing_column():
    df = pd.DataFrame({"Item": ["kiwi"]})

    with pytest.raises(KeyError, match="Column not found"):
        count_rows_starting_with(df, "Name", "k")


def test_count_rows_starting_with_ignores_text_in_middle():
    df = pd.DataFrame({"Item": ["apple pie", "pineapple"]})

    assert count_rows_starting_with(df, "Item", "apple") == 1
