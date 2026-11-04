import pytest

from zfinance.normalize import merchant_key, normalize_description


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("TIM HORTONS #4521 LONDON ON", "tim hortons london"),
        ("SQ *GOLDEN BAKERY 0423", "golden bakery"),
        ("POS PURCHASE 10/14 METRO 512", "metro"),
        ("POS PURCHASE SQ *GOLDEN BAKERY", "golden bakery"),
        ("VISA DEBIT PURCHASE XX1234 SHELL", "shell"),
        ("MCDONALD'S #7326", "mcdonald's"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_description(raw, expected):
    assert normalize_description(raw) == expected


def test_same_merchant_normalizes_identically():
    a = normalize_description("TIM HORTONS #4521 LONDON ON")
    b = normalize_description("CONTACTLESS TIM HORTONS #88 LONDON ON")
    assert a == b


def test_merchant_key_first_two_tokens():
    assert merchant_key("second cup st thomas") == "second cup"
    assert merchant_key("metro") == "metro"
