import pandas as pd
import pytest

from data_analysis import fill_missing_with_previous


def test_fill_missing_with_previous():
    df = pd.DataFrame({"Stock": [1, None, 3]})

    assert fill_missing_with_previous(df, "Stock") == [1, 1, 3]


def test_fill_missing_with_previous_leading_missing():
    df = pd.DataFrame({"Stock": [None, 2, None]})

    assert fill_missing_with_previous(df, "Stock") == [None, 2, 2]


def test_fill_missing_with_previous_missing_column():
    df = pd.DataFrame({"Stock": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        fill_missing_with_previous(df, "Sales")


def test_fill_missing_with_previous_several_gaps():
    df = pd.DataFrame({"Stock": [1, None, None]})

    assert fill_missing_with_previous(df, "Stock") == [1, 1, 1]
