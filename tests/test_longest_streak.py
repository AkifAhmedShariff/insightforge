import pandas as pd

from data_analysis import get_longest_streak_above


def test_longest_streak_above():
    df = pd.DataFrame({"Sales": [5, 1]})

    assert get_longest_streak_above(df, "Sales", 2) == 1


def test_longest_streak_above_none_above():
    df = pd.DataFrame({"Sales": [1, 2]})

    assert get_longest_streak_above(df, "Sales", 2) == 0


def test_longest_streak_above_missing_value_ends_streak():
    df = pd.DataFrame({"Sales": [5, None, 5, 1]})

    assert get_longest_streak_above(df, "Sales", 2) == 1


def test_longest_streak_above_counts_streak_at_end():
    df = pd.DataFrame({"Sales": [1, 5, 5]})

    assert get_longest_streak_above(df, "Sales", 2) == 2
