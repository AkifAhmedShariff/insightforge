import pandas as pd
import pytest

from data_analysis import get_second_highest_value


def test_second_highest_value():
    df = pd.DataFrame({"Sales": [3, 9, 5]})

    assert get_second_highest_value(df, "Sales") == 5.0


def test_second_highest_value_skips_missing_values():
    df = pd.DataFrame({"Sales": [7, None, 2]})

    assert get_second_highest_value(df, "Sales") == 2.0


def test_second_highest_value_needs_two_distinct_values():
    df = pd.DataFrame({"Sales": [4, 4]})

    with pytest.raises(ValueError, match="two distinct values"):
        get_second_highest_value(df, "Sales")


def test_second_highest_value_ignores_repeated_highest():
    df = pd.DataFrame({"Sales": [9, 9, 5]})

    assert get_second_highest_value(df, "Sales") == 5.0
