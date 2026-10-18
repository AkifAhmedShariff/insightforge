import pandas as pd
import pytest

from data_analysis import get_unique_count


def test_unique_count():
    df = pd.DataFrame({"Region": ["North", "South", "North"]})

    assert get_unique_count(df, "Region") == 2


def test_unique_count_missing_column():
    df = pd.DataFrame({"Region": ["North"]})

    with pytest.raises(KeyError, match="Column not found"):
        get_unique_count(df, "City")
