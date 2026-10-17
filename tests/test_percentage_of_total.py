import pandas as pd

from data_analysis import get_percentage_of_total


def test_percentage_of_total():
    df = pd.DataFrame({"Sales": [0, 100]})

    assert get_percentage_of_total(df, "Sales") == [0.0, 100.0]


def test_percentage_of_total_all_zero():
    df = pd.DataFrame({"Sales": [0, 0]})

    assert get_percentage_of_total(df, "Sales") == [0.0, 0.0]
