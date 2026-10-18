import pandas as pd

from data_analysis import convert_to_numeric


def test_convert_to_numeric():
    df = pd.DataFrame({"Sales": ["10", "2.5", "abc"]})

    assert convert_to_numeric(df, "Sales") == [10.0, 2.5, None]
