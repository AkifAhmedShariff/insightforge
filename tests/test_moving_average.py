import pandas as pd
import pytest

from data_analysis import get_moving_average


def test_moving_average_returns_one_value_per_row():
    df = pd.DataFrame({"Sales": [10, 20, 30, 40, 50]})

    result = get_moving_average(df, "Sales")

    assert len(result) == 5
    assert result[-1] is not None


def test_moving_average_rejects_zero_window():
    df = pd.DataFrame({"Sales": [10, 20]})

    with pytest.raises(ValueError, match="at least 1"):
        get_moving_average(df, "Sales", window=0)
