"""Clean raw bank descriptions into something a model can learn from.

Bank descriptions are noisy: "TIM HORTONS #4521 LONDON ON", "SQ *GOLDEN BAKERY 0423",
"POS PURCHASE 10/14 METRO 512". The same merchant shows up dozens of ways. Normalizing
strips the noise that changes per-transaction (store numbers, dates, card digits,
location suffixes) while keeping the merchant signal.
"""

from __future__ import annotations

import re

# Prefixes added by payment processors / banks that carry no merchant signal.
_PREFIXES = re.compile(
    r"^(POS PURCHASE|POS|PURCHASE|DEBIT PURCHASE|INTERAC PURCHASE|IDP PURCHASE|"
    r"VISA DEBIT PURCHASE|CONTACTLESS|PRE-AUTH|PREAUTH|PAYMENT|"
    r"SQ \*|SQ\*|TST\*|TST \*|PAYPAL \*|PP\*|SP \*|SP\*)\s*",
    re.IGNORECASE,
)

_PROVINCES = r"(ON|QC|BC|AB|MB|SK|NS|NB|NL|PE|YT|NT|NU)"

_PATTERNS = [
    (re.compile(r"https?://\S+|www\.\S+", re.I), " "),
    (re.compile(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b"), " "),  # dates
    (re.compile(r"#\s*\d+"), " "),  # store numbers "#4521"
    (re.compile(r"\b[xX*]{2,}\d{2,4}\b"), " "),  # masked card numbers
    (re.compile(r"\b\d{3,}\b"), " "),  # long digit runs (terminal ids, refs)
    (re.compile(rf"\b{_PROVINCES}\b\s*$"), " "),  # trailing province code
    (re.compile(r"[^\w\s&'.]"), " "),  # punctuation except a few merchant-name chars
    (re.compile(r"\s+"), " "),
]


def normalize_description(raw: str) -> str:
    """Return a lowercase, de-noised merchant string."""
    if raw is None:
        return ""
    s = str(raw).strip()
    # Strip stacked prefixes ("POS PURCHASE SQ *...")
    for _ in range(3):
        new = _PREFIXES.sub("", s)
        if new == s:
            break
        s = new
    for pattern, repl in _PATTERNS:
        s = pattern.sub(repl, s)
    return s.strip().lower()


def merchant_key(clean_desc: str) -> str:
    """Coarse merchant identity: the first two tokens of a cleaned description.

    "second cup london" and "second cup st thomas" are the same merchant. Used to group
    cross-validation folds so evaluation never tests on a merchant seen in training.
    """
    return " ".join(clean_desc.split()[:2])
