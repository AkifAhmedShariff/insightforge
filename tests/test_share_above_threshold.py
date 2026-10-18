import pandas as pd

from data_analysis import get_share_above_threshold


def test_share_above_threshold():
    df = pd.DataFrame({"Sales": [50, 150, 250, 80]})

    assert get_share_above_threshold(df, "Sales", 100) == 50.0


def test_share_above_threshold_empty_column():
    df = pd.DataFrame({"Sales": [None, None]})

    assert get_share_above_threshold(df, "Sales", 100) == 0.0
