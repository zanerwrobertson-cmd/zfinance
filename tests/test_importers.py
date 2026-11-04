from pathlib import Path

import pytest

from zfinance.importers import ImportFormatError, parse_csv, parse_mapped

SAMPLES = Path(__file__).resolve().parent.parent / "sample_data"

RBC = """Account Type,Account Number,Transaction Date,Cheque Number,Description 1,Description 2,CAD$,USD$
Chequing,01234-5678901,10/01/2025,,PREAUTHORIZED DEBIT NORTHSIDE PROPERTY MGMT,,-875.00,
Chequing,01234-5678901,10/02/2025,,MCDONALD'S #7326,,-10.06,
Chequing,01234-5678901,10/15/2025,,PAYROLL DEP,ACME CORP,1200.50,
"""

TD = """10/01/2025,PREAUTHORIZED DEBIT NORTHSIDE PROPERTY MGMT,875.00,,1625.00
10/02/2025,MCDONALD'S #7326,10.06,,1614.94
10/15/2025,PAYROLL DEP,,1200.50,2815.44
"""

HEADERLESS_SIGNED = """2025-10-01,COFFEE SHOP,-4.50
2025-10-02,GROCERY STORE,-62.10
2025-10-03,PAYROLL,900.00
"""


def test_rbc_header_format():
    res = parse_csv(RBC)
    df = res.df
    assert len(df) == 3
    assert list(df["amount"]) == [-875.0, -10.06, 1200.5]
    assert df["date"].iloc[0].strftime("%Y-%m-%d") == "2025-10-01"
    # Description 2 is appended to Description 1
    assert df["description"].iloc[2] == "PAYROLL DEP ACME CORP"


def test_td_headerless_debit_credit():
    res = parse_csv(TD)
    assert res.source_format.startswith("Debit/Credit")
    # debit -> negative, credit -> positive, balance column ignored
    assert list(res.df["amount"]) == [-875.0, -10.06, 1200.5]


def test_generic_headerless_signed_amount():
    res = parse_csv(HEADERLESS_SIGNED)
    assert res.source_format == "Generic (headerless)"
    assert list(res.df["description"]) == ["COFFEE SHOP", "GROCERY STORE", "PAYROLL"]
    assert list(res.df["amount"]) == [-4.5, -62.1, 900.0]


def test_invert_sign_and_account_name():
    res = parse_csv(HEADERLESS_SIGNED, account_name="Visa", invert_sign=True)
    assert list(res.df["amount"]) == [4.5, 62.1, -900.0]
    assert set(res.df["account"]) == {"Visa"}


def test_sample_files_parse_to_same_amounts():
    rbc = parse_csv((SAMPLES / "rbc_export_example.csv").read_bytes()).df
    td = parse_csv((SAMPLES / "td_export_example.csv").read_bytes()).df
    assert len(rbc) == len(td)
    assert rbc["amount"].round(2).tolist() == td["amount"].round(2).tolist()


def test_unparseable_raises_format_error():
    with pytest.raises(ImportFormatError):
        parse_csv("foo,bar\nbaz,qux\n")


def test_manual_mapping_fallback():
    data = "junk line\nhello,2025-10-01,-3.25\nworld,2025-10-02,7.00\n"
    res = parse_mapped(data, date_col=1, desc_col=0, amount_col=2, skip_rows=1)
    assert list(res.df["amount"]) == [-3.25, 7.0]
    assert list(res.df["description"]) == ["hello", "world"]
