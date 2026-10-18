import pandas as pd

from data_analysis import get_column_ratios


def test_column_ratios_returns_one_value_per_row():
    df = pd.DataFrame({
        "Profit": [10, 20],
        "Sales": [100, 200],
    })

    result = get_column_ratios(df, "Profit", "Sales")

    assert len(result) == 2
    assert all(value is not None for value in result)


def test_column_ratios_missing_value_is_none():
    df = pd.DataFrame({"Profit": [None], "Sales": [100]})

    assert get_column_ratios(df, "Profit", "Sales") == [None]


def test_column_ratios_divide_by_denominator():
    df = pd.DataFrame({
        "Profit": [10, 20, 5],
        "Sales": [100, 0, 50],
    })

    assert get_column_ratios(df, "Profit", "Sales") == [0.1, None, 0.1]
