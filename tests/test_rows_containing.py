import pandas as pd
import pytest

from data_analysis import filter_rows_containing


def test_filter_rows_containing_ignores_case():
    df = pd.DataFrame({"Product": ["Blue Pen", "Red Mug"]})

    result = filter_rows_containing(df, "Product", "pen")

    assert result["Product"].tolist() == ["Blue Pen"]


def test_filter_rows_containing_skips_missing_values():
    df = pd.DataFrame({"Product": [None, "Pen"]})

    result = filter_rows_containing(df, "Product", "Pen")

    assert result["Product"].tolist() == ["Pen"]


def test_filter_rows_containing_missing_column():
    df = pd.DataFrame({"Product": ["Pen"]})

    with pytest.raises(KeyError, match="Column not found"):
        filter_rows_containing(df, "Region", "Pen")


def test_filter_rows_containing_matches_text_literally():
    df = pd.DataFrame({"Product": ["Model A.1", "Model AX1"]})

    result = filter_rows_containing(df, "Product", "A.1")

    assert result["Product"].tolist() == ["Model A.1"]
