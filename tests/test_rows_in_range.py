import pandas as pd
import pytest

from data_analysis import get_rows_in_range


def test_rows_in_range_filters_values():
    df = pd.DataFrame({
        "Product": ["A", "B", "C", "D"],
        "Sales": [50, 120, 180, 300],
    })

    result = get_rows_in_range(df, "Sales", 100, 200)

    assert list(result["Product"]) == ["B", "C"]


def test_rows_in_range_rejects_reversed_bounds():
    df = pd.DataFrame({"Sales": [10, 20]})

    with pytest.raises(ValueError, match="greater than high"):
        get_rows_in_range(df, "Sales", 200, 100)
