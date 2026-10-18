import pandas as pd

from data_analysis import get_ranks


def test_ranks_ties_share_rank():
    df = pd.DataFrame({"Sales": [50, 50]})

    assert get_ranks(df, "Sales") == [1, 1]


def test_ranks_missing_value_is_none():
    df = pd.DataFrame({"Sales": [None, 10]})

    assert get_ranks(df, "Sales") == [None, 1]


def test_ranks_highest_value_is_rank_one():
    df = pd.DataFrame({"Sales": [100, 300, 200]})

    assert get_ranks(df, "Sales") == [3, 1, 2]
