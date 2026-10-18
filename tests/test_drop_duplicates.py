import pandas as pd
import pytest

from data_analysis import drop_duplicate_rows


def test_drop_duplicate_rows():
    df = pd.DataFrame({
        "Product": ["A", "A", "B"],
        "Sales": [100, 100, 200],
    })

    result = drop_duplicate_rows(df)

    assert result.to_dict("records") == [
        {"Product": "A", "Sales": 100},
        {"Product": "B", "Sales": 200},
    ]


def test_drop_duplicate_rows_unknown_subset_column():
    df = pd.DataFrame({"Product": ["A"]})

    with pytest.raises(KeyError, match="Column not found"):
        drop_duplicate_rows(df, subset=["Region"])


def test_drop_duplicate_rows_keeps_first_occurrence_in_subset():
    df = pd.DataFrame({
        "Product": ["A", "A"],
        "Sales": [100, 999],
    })

    result = drop_duplicate_rows(df, subset=["Product"])

    assert result.to_dict("records") == [{"Product": "A", "Sales": 100}]
