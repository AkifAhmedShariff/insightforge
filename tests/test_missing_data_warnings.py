import pandas as pd

from data_analysis import collect_missing_data_warnings


def test_collect_missing_data_warnings():
    df = pd.DataFrame({
        "Sales": [None, None, 3],
        "Region": ["North", "South", None],
    })

    assert collect_missing_data_warnings(df) == ["Sales: 67% missing"]
