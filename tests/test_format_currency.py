from data_analysis import format_currency


def test_format_currency():
    assert format_currency(1234.5) == "$1,234.50"


def test_format_currency_custom_symbol_and_missing():
    assert format_currency(99, symbol="₹") == "₹99.00"
    assert format_currency(None) == ""
