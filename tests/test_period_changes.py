import pandas as pd

from data_analysis import get_period_changes


def test_period_changes_first_row_is_none():
    df = pd.DataFrame({"Sales": [100, 100]})

    assert get_period_changes(df, "Sales") == [None, 0.0]


def test_period_changes_length_matches_rows():
    df = pd.DataFrame({"Sales": [1, 2, 3, 4]})

    assert len(get_period_changes(df, "Sales")) == 4


def test_period_changes_are_percentages():
    df = pd.DataFrame({"Sales": [100, 150, 75]})

    assert get_period_changes(df, "Sales") == [None, 50.0, -50.0]
