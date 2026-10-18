import pandas as pd
import pytest

from data_analysis import get_true_percentage


def test_true_percentage():
    df = pd.DataFrame({"Paid": [True, False, True, True]})

    assert get_true_percentage(df, "Paid") == 75.0


def test_true_percentage_all_false():
    df = pd.DataFrame({"Paid": [False, False]})

    assert get_true_percentage(df, "Paid") == 0.0


def test_true_percentage_rejects_empty_column():
    df = pd.DataFrame({"Paid": [None]})

    with pytest.raises(ValueError, match="no values"):
        get_true_percentage(df, "Paid")


def test_true_percentage_missing_column():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(KeyError, match="Column not found"):
        get_true_percentage(df, "Paid")


def test_true_percentage_skips_missing_values():
    df = pd.DataFrame({"Paid": [True, None, False]})

    assert get_true_percentage(df, "Paid") == 50.0
