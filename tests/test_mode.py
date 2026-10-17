import pandas as pd
import pytest

from data_analysis import get_mode


def test_mode_of_repeated_value():
    df = pd.DataFrame({"Region": ["North", "North"]})

    assert get_mode(df, "Region") == "North"


def test_mode_rejects_empty_column():
    df = pd.DataFrame({"Region": [None, None]})

    with pytest.raises(ValueError, match="no values"):
        get_mode(df, "Region")
