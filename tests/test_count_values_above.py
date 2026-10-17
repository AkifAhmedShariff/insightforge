import pandas as pd

from data_analysis import count_values_above


def test_count_values_above_threshold():
    df = pd.DataFrame({"Sales": [50, 120, 180, 300]})

    assert count_values_above(df, "Sales", 100) == 3


def test_count_values_above_ignores_missing():
    df = pd.DataFrame({"Sales": [None, 500]})

    assert count_values_above(df, "Sales", 100) == 1


def test_count_values_above_excludes_threshold():
    df = pd.DataFrame({"Sales": [100, 101]})

    assert count_values_above(df, "Sales", 100) == 1
