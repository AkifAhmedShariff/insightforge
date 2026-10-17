import pandas as pd

from data_analysis import normalize_column


def test_normalize_column_scales_to_unit_range():
    df = pd.DataFrame({"Sales": [0, 5, 10]})

    assert normalize_column(df, "Sales") == [0.0, 0.5, 1.0]


def test_normalize_constant_column_is_zero():
    df = pd.DataFrame({"Sales": [7, 7]})

    assert normalize_column(df, "Sales") == [0.0, 0.0]
