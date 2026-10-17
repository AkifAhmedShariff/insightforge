import pandas as pd

from data_analysis import get_z_scores


def test_z_scores_are_centered_on_mean():
    df = pd.DataFrame({"Sales": [1, 2, 3]})

    assert get_z_scores(df, "Sales") == [-1.0, 0.0, 1.0]


def test_z_scores_of_constant_column_are_zero():
    df = pd.DataFrame({"Sales": [5, 5, 5]})

    assert get_z_scores(df, "Sales") == [0.0, 0.0, 0.0]
