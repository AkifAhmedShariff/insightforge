import pandas as pd
import pytest

from data_analysis import get_top_n_rows


def test_top_n_rows_sorted_largest_first():
    df = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Sales": [100, 300, 200],
    })

    result = get_top_n_rows(df, "Sales")

    assert list(result["Product"]) == ["B", "C", "A"]


def test_top_n_rows_rejects_zero():
    df = pd.DataFrame({"Sales": [10, 20]})

    with pytest.raises(ValueError, match="at least 1"):
        get_top_n_rows(df, "Sales", n=0)


def test_top_n_rows_returns_exactly_n_rows():
    df = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Sales": [100, 300, 200],
    })

    result = get_top_n_rows(df, "Sales", n=2)

    assert list(result["Product"]) == ["B", "C"]
