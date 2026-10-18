import pandas as pd

from data_analysis import get_cumulative_sum


def test_cumulative_sum():
    df = pd.DataFrame({"Sales": [0, 0, 5]})

    assert get_cumulative_sum(df, "Sales") == [0.0, 0.0, 5.0]


def test_cumulative_sum_treats_missing_as_zero():
    df = pd.DataFrame({"Sales": [None, 7]})

    assert get_cumulative_sum(df, "Sales") == [0.0, 7.0]
