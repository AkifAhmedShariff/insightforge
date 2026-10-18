import pandas as pd

from data_analysis import count_missing_by_row


def test_count_missing_by_row():
    df = pd.DataFrame({"A": [None, 1], "B": [1, None]})

    assert count_missing_by_row(df) == [1, 1]


def test_count_missing_by_row_no_missing():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})

    assert count_missing_by_row(df) == [0, 0]
