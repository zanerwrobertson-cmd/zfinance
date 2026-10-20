"""Parse bank CSV exports into one normalized schema.

Output columns: date (datetime64), description (str), amount (float, negative = money out),
account (str), source_format (str).

Canadian banks each export differently: some have headers (RBC, BMO, newer Scotiabank),
some don't (TD, CIBC), some prepend junk lines (BMO). Rather than hard-code every format,
this module does two passes:

1. Header detection: scan the first rows for a line that looks like a header and map
   known column names (date / description / amount / debit / credit) to the schema.
2. Headerless inference: if no header is found, infer columns by content — which column
   parses as dates, which is mostly letters, which are numeric — and whether two numeric
   columns behave like debit/credit (mutually exclusive per row).

Anything that still can't be parsed raises ImportFormatError with a useful message, and
the app falls back to manual column mapping.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

import pandas as pd


class ImportFormatError(ValueError):
    pass


# Header synonyms, lowercase. Order matters: first match wins.
DATE_HEADERS = ["transaction date", "date posted", "posted date", "date"]
DESC_HEADERS = ["description 1", "description", "merchant", "payee", "details", "name", "memo"]
DESC2_HEADERS = ["description 2", "sub-description", "sub description"]
AMOUNT_HEADERS = ["cad$", "transaction amount", "amount", "cad amount"]
DEBIT_HEADERS = ["debit", "withdrawals", "withdrawal", "money out"]
CREDIT_HEADERS = ["credit", "deposits", "deposit", "money in"]
ACCOUNT_HEADERS = ["account type", "account", "first bank card", "card number"]


@dataclass
class ParseResult:
    df: pd.DataFrame
    source_format: str
    warnings: list[str]


def _read_raw(data: bytes | str) -> list[list[str]]:
    if isinstance(data, bytes):
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
    else:
        text = data
    rows = list(csv.reader(io.StringIO(text)))
    rows = [[c.strip() for c in r] for r in rows]
    return [r for r in rows if any(c for c in r)]


def _parse_amount(val) -> float | None:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s == "-":
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.strip("()").replace("$", "").replace(",", "").replace(" ", "")
    if s.upper().endswith("CR"):
        s = s[:-2]
    try:
        v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def _parse_dates(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    # Compact YYYYMMDD (BMO)
    if s.str.fullmatch(r"\d{8}").mean() > 0.8:
        return pd.to_datetime(s, format="%Y%m%d", errors="coerce")
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%Y", "%b %d, %Y"):
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().mean() > 0.8:
            return parsed
    return pd.to_datetime(s, errors="coerce")


def _date_ratio(values: list[str]) -> float:
    vals = [v for v in values if v]
    if not vals:
        return 0.0
    return float(_parse_dates(pd.Series(vals)).notna().mean())


def _numeric_ratio(values: list[str]) -> float:
    vals = [v for v in values if v]
    if not vals:
        return 0.0
    return sum(_parse_amount(v) is not None for v in vals) / len(vals)


def _alpha_score(values: list[str]) -> float:
    vals = [v for v in values if v]
    if not vals:
        return 0.0
    return sum(len(re.findall(r"[A-Za-z]", v)) for v in vals) / len(vals)


def _find(headers: list[str], candidates: list[str]) -> int | None:
    for cand in candidates:
        for i, h in enumerate(headers):
            if h == cand:
                return i
    for cand in candidates:
        for i, h in enumerate(headers):
            if cand in h:
                return i
    return None


def _guess_format(headers: list[str]) -> str:
    joined = " | ".join(headers)
    if "cad$" in joined and "description 1" in joined:
        return "RBC"
    if "first bank card" in joined or "date posted" in joined:
        return "BMO"
    if "sub-description" in joined or "type of transaction" in joined:
        return "Scotiabank"
    return "Generic (header)"


def _from_header(rows: list[list[str]], header_idx: int) -> ParseResult | None:
    headers = [h.lower().strip() for h in rows[header_idx]]
    body = rows[header_idx + 1 :]
    width = len(headers)
    body = [(r + [""] * width)[:width] for r in body]

    i_date = _find(headers, DATE_HEADERS)
    i_desc = _find(headers, DESC_HEADERS)
    i_desc2 = _find(headers, DESC2_HEADERS)
    i_amt = _find(headers, AMOUNT_HEADERS)
    i_deb = _find(headers, DEBIT_HEADERS)
    i_cred = _find(headers, CREDIT_HEADERS)
    i_acct = _find(headers, ACCOUNT_HEADERS)

    if i_date is None or i_desc is None or (i_amt is None and i_deb is None):
        return None
    if i_desc2 == i_desc:
        i_desc2 = None

    fmt = _guess_format(headers)
    warnings: list[str] = []
    out = pd.DataFrame()
    out["date"] = _parse_dates(pd.Series([r[i_date] for r in body]))
    desc = [r[i_desc] for r in body]
    if i_desc2 is not None:
        desc = [f"{a} {r[i_desc2]}".strip() for a, r in zip(desc, body)]
    out["description"] = desc

    if i_amt is not None:
        out["amount"] = [_parse_amount(r[i_amt]) for r in body]
    else:
        deb = [_parse_amount(r[i_deb]) or 0.0 for r in body]
        cred = [(_parse_amount(r[i_cred]) or 0.0) if i_cred is not None else 0.0 for r in body]
        out["amount"] = [c - abs(d) for d, c in zip(deb, cred)]

    out["account"] = [r[i_acct] for r in body] if i_acct is not None else ""
    return ParseResult(out, fmt, warnings)


def _from_inference(rows: list[list[str]]) -> ParseResult:
    width = max(len(r) for r in rows)
    rows = [(r + [""] * width)[:width] for r in rows]
    cols = list(zip(*rows))

    date_scores = [_date_ratio(list(c)) for c in cols]
    i_date = max(range(width), key=lambda i: date_scores[i])
    if date_scores[i_date] < 0.8:
        raise ImportFormatError("Couldn't find a date column. Map columns manually.")

    numeric = [
        i
        for i in range(width)
        if i != i_date and _numeric_ratio(list(cols[i])) > 0.9 and any(cols[i])
    ]
    text_cols = [i for i in range(width) if i != i_date and i not in numeric]
    if not text_cols:
        raise ImportFormatError("Couldn't find a description column. Map columns manually.")
    i_desc = max(text_cols, key=lambda i: _alpha_score(list(cols[i])))

    # Ignore numeric columns that look like masked card numbers (CIBC appends these).
    numeric = [i for i in numeric if not all(re.fullmatch(r"[\d*xX]{12,19}", v) or not v for v in cols[i])]
    if not numeric:
        raise ImportFormatError("Couldn't find an amount column. Map columns manually.")

    warnings: list[str] = []
    fmt = "Generic (headerless)"
    amounts: list[float | None]
    if len(numeric) >= 2:
        a, b = numeric[0], numeric[1]
        both = sum(1 for r in rows if r[a] and r[b])
        if both / len(rows) < 0.05:
            # Debit/credit pair (TD, CIBC). Any third numeric column is a running balance.
            fmt = "Debit/Credit (TD/CIBC style)"
            amounts = [
                (_parse_amount(r[b]) or 0.0) - abs(_parse_amount(r[a]) or 0.0) for r in rows
            ]
        else:
            amounts = [_parse_amount(r[a]) for r in rows]
            warnings.append("Multiple numeric columns; used the first as the amount.")
    else:
        amounts = [_parse_amount(r[numeric[0]]) for r in rows]

    out = pd.DataFrame(
        {
            "date": _parse_dates(pd.Series(cols[i_date])),
            "description": list(cols[i_desc]),
            "amount": amounts,
            "account": "",
        }
    )
    return ParseResult(out, fmt, warnings)


def parse_csv(data: bytes | str, account_name: str = "", invert_sign: bool = False) -> ParseResult:
    """Parse a bank CSV export. Negative amounts are money out."""
    rows = _read_raw(data)
    if not rows:
        raise ImportFormatError("File is empty.")

    result = None
    for idx, row in enumerate(rows[:15]):
        lowered = [c.lower() for c in row]
        if any(h in c for c in lowered for h in DATE_HEADERS) and (
            _date_ratio(row) < 0.5
        ):
            result = _from_header(rows, idx)
            if result is not None:
                break
    if result is None:
        # Drop preamble lines (fewer cells than the data body) before inferring.
        body_width = pd.Series([len(r) for r in rows]).mode().iloc[0]
        result = _from_inference([r for r in rows if len(r) >= body_width])

    df = result.df
    before = len(df)
    df = df.dropna(subset=["date", "amount"])
    df = df[df["description"].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        result.warnings.append(f"Skipped {dropped} row(s) with a missing date, amount or description.")
    if df.empty:
        raise ImportFormatError("No valid transactions found in this file.")

    if invert_sign:
        df["amount"] = -df["amount"]
    if account_name:
        df["account"] = account_name
    df["description"] = df["description"].astype(str).str.strip()
    df["amount"] = df["amount"].astype(float).round(2)
    df["source_format"] = result.source_format
    result.df = df.reset_index(drop=True)
    return result


def parse_mapped(
    data: bytes | str,
    date_col: int,
    desc_col: int,
    amount_col: int,
    skip_rows: int = 0,
    account_name: str = "",
    invert_sign: bool = False,
) -> ParseResult:
    """Manual fallback: the user says which column is which."""
    rows = _read_raw(data)[skip_rows:]
    width = max(len(r) for r in rows)
    rows = [(r + [""] * width)[:width] for r in rows]
    df = pd.DataFrame(
        {
            "date": _parse_dates(pd.Series([r[date_col] for r in rows])),
            "description": [r[desc_col] for r in rows],
            "amount": [_parse_amount(r[amount_col]) for r in rows],
            "account": account_name,
        }
    ).dropna(subset=["date", "amount"])
    if invert_sign:
        df["amount"] = -df["amount"]
    df["source_format"] = "Manual mapping"
    return ParseResult(df.reset_index(drop=True), "Manual mapping", [])


def preview_rows(data: bytes | str, n: int = 8) -> list[list[str]]:
    return _read_raw(data)[:n]
