import pandas as pd
import pytest

from data_analysis import get_page


def test_get_page_returns_rows():
    df = pd.DataFrame({"Sales": [1, 2, 3]})

    assert list(get_page(df, 1)["Sales"]) == [1, 2, 3]


def test_get_page_rejects_page_zero():
    df = pd.DataFrame({"Sales": [1]})

    with pytest.raises(ValueError, match="at least 1"):
        get_page(df, 0)
