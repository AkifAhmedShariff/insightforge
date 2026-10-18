import pandas as pd
import pytest

from data_analysis import get_weighted_average


def test_weighted_average():
    df = pd.DataFrame({"Price": [10, 20], "Units": [1, 1]})

    assert get_weighted_average(df, "Price", "Units") == 15.0


def test_weighted_average_rejects_zero_weights():
    df = pd.DataFrame({"Price": [10, 20], "Units": [0, 0]})

    with pytest.raises(ValueError, match="non-zero total"):
        get_weighted_average(df, "Price", "Units")


def test_weighted_average_uses_weight_total():
    df = pd.DataFrame({"Price": [10, 20], "Units": [1, 3]})

    assert get_weighted_average(df, "Price", "Units") == 17.5
