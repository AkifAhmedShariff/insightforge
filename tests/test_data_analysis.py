
import pandas as pd
import pytest

from data_analysis import (
    get_data_quality_summary,
    get_numeric_statistics,
    get_highest_value_row,
)


def test_data_quality_summary_counts_missing_and_duplicate_rows():
    df = pd.DataFrame({
        "Product": ["A", "B", "A"],
        "Sales": [100, None, 100],
    })

    result = get_data_quality_summary(df)

    assert result == {
        "row_count": 3,
        "column_count": 2,
        "missing_value_count": 1,
        "duplicate_row_count": 1,
    }


def test_numeric_statistics():
    df = pd.DataFrame({"Sales": [10, 20, 30]})

    result = get_numeric_statistics(df, "Sales")

    assert result == {
        "count": 3,
        "mean": 20.0,
        "median": 20.0,
        "min": 10.0,
        "max": 30.0,
    }


def test_numeric_statistics_ignores_missing_values():
    df = pd.DataFrame({"Sales": [10, None, 30]})

    result = get_numeric_statistics(df, "Sales")

    assert result["count"] == 2
    assert result["mean"] == 20.0


def test_empty_numeric_column_returns_empty_statistics():
    df = pd.DataFrame({"Sales": [None, None]})

    result = get_numeric_statistics(df, "Sales")

    assert result == {
        "count": 0,
        "mean": None,
        "median": None,
        "min": None,
        "max": None,
    }


def test_missing_column_raises_clear_error():
    df = pd.DataFrame({"Sales": [10, 20]})

    with pytest.raises(KeyError, match="Column not found"):
        get_numeric_statistics(df, "Revenue")


def test_non_numeric_column_is_rejected():
    df = pd.DataFrame({"Product": ["A", "B"]})

    with pytest.raises(TypeError, match="Column is not numeric"):
        get_numeric_statistics(df, "Product")


def test_highest_value_returns_correct_record():
    df = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Sales": [100, 500, 250],
    })

    result = get_highest_value_row(df, "Sales")

    assert result["Product"] == "B"
    assert result["Sales"] == 500


def test_highest_value_ignores_missing_values():
    df = pd.DataFrame({
        "Product": ["A", "B"],
        "Sales": [None, 250],
    })

    result = get_highest_value_row(df, "Sales")

    assert result["Product"] == "B"
    assert result["Sales"] == 250


def test_highest_value_rejects_all_missing_column():
    df = pd.DataFrame({"Sales": [None, None]})

    with pytest.raises(ValueError, match="no numeric values"):
        get_highest_value_row(df, "Sales")
