"""Synthetic transaction generator.

Produces a year of realistic, messy, labelled transactions for a student in London, ON,
so the repo runs end-to-end without anyone's real banking data. Descriptions are
deliberately noisy (processor prefixes, store numbers, city/province suffixes, dates)
because that noise is what the normalizer and model have to cope with on real exports.

Also writes the same data in RBC-style and TD-style CSV layouts to exercise the importer.
"""

from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# category -> list of (merchant, low, high)
VARIABLE = {
    "Groceries": [
        ("METRO", 18, 95), ("NO FRILLS", 15, 110), ("LOBLAWS", 20, 120), ("FOOD BASICS", 12, 85),
        ("FRESHCO", 15, 90), ("COSTCO WHOLESALE", 60, 220), ("SOBEYS", 15, 80), ("FARM BOY", 12, 70),
        ("BULK BARN", 6, 35), ("T&T SUPERMARKET", 20, 90), ("REMARK FRESH MARKETS", 15, 75),
        ("COVENT GARDEN MARKET", 8, 40), ("STARSKY FINE FOODS", 12, 60), ("ZEHRS MARKETS", 18, 95),
    ],
    "Dining": [
        ("MCDONALD'S", 6, 18), ("SUBWAY", 8, 16), ("A&W", 9, 17), ("HARVEY'S", 9, 18),
        ("PIZZA PIZZA", 12, 35), ("DOMINO'S PIZZA", 15, 40), ("UBER EATS", 18, 45), ("DOORDASH", 18, 50),
        ("SKIPTHEDISHES", 17, 45), ("OSMOW'S SHAWARMA", 12, 22), ("MARY BROWN'S CHICKEN", 10, 25),
        ("CHIPOTLE MEXICAN GRILL", 12, 20), ("THAI TASTE RESTAURANT", 15, 40), ("BURRITO BOYZ", 12, 20),
        ("LAZEEZ SHAWARMA", 11, 20), ("BARBURRITO", 12, 19), ("BOSTON PIZZA", 22, 60),
        ("EAST SIDE MARIO'S", 25, 65), ("THE CEEPS BAR", 15, 55), ("JOE KOOLS", 18, 60),
        ("MUCHO BURRITO", 12, 19), ("WENDY'S", 8, 16), ("POPEYES LOUISIANA KITCHEN", 9, 20),
    ],
    "Coffee": [
        ("TIM HORTONS", 2.5, 9), ("STARBUCKS", 5, 12), ("SECOND CUP", 4, 10), ("BALZAC'S COFFEE", 4, 11),
        ("LOCOMOTIVE ESPRESSO", 4, 10), ("WILLIAMS FRESH CAFE", 4, 12), ("COFFEE CULTURE CAFE", 4, 11),
        ("BLACK WALNUT BAKERY CAFE", 4, 13),
    ],
    "Transport": [
        ("LONDON TRANSIT COMMISSION", 3.2, 3.2), ("UBER TRIP", 9, 28), ("LYFT RIDE", 9, 26),
        ("PETRO-CANADA", 30, 70), ("ESSO", 30, 70), ("SHELL", 30, 70), ("PIONEER GAS", 30, 65),
        ("CANADIAN TIRE GAS BAR", 30, 65), ("HONK MOBILE PARKING", 3, 12), ("IMPARK PARKING", 4, 15),
    ],
    "Shopping": [
        ("AMAZON.CA", 12, 120), ("CANADIAN TIRE", 10, 140), ("BEST BUY", 20, 300), ("WINNERS", 15, 90),
        ("H&M", 15, 80), ("UNIQLO", 20, 110), ("SPORT CHEK", 25, 160), ("INDIGO BOOKS", 12, 60),
        ("DOLLARAMA", 3, 25), ("IKEA", 20, 250), ("STAPLES", 8, 60), ("MARK'S WORK WEARHOUSE", 25, 120),
        ("WALMART SUPERCENTRE", 10, 90), ("SHEIN", 20, 70),
    ],
    "Entertainment": [
        ("CINEPLEX ENTERTAINMENT", 13, 35), ("STEAM GAMES", 10, 80), ("PLAYSTATION NETWORK", 10, 90),
        ("TICKETMASTER", 40, 180), ("BUDWEISER GARDENS", 30, 120), ("NINTENDO ESHOP", 8, 80),
        ("PLAYDIUM", 20, 60), ("JUNCTION CLIMBING CENTRE", 18, 25),
    ],
    "Health": [
        ("SHOPPERS DRUG MART", 6, 60), ("REXALL PHARMACY", 6, 50), ("MASONVILLE DENTAL", 80, 220),
        ("FOREST CITY PHYSIOTHERAPY", 60, 110), ("IRIS OPTOMETRY", 50, 200),
    ],
    "Fees": [
        ("MONTHLY ACCOUNT FEE", 4.95, 4.95), ("INTERAC E-TRANSFER FEE", 1.5, 1.5),
        ("NSF SERVICE CHARGE", 48, 48), ("OVERDRAFT INTEREST", 1, 9), ("FOREIGN TRANSACTION FEE", 0.5, 4),
    ],
    "Other": [
        ("ATM WITHDRAWAL", 20, 100), ("CANADA POST", 3, 30), ("SERVICEONTARIO", 32, 120),
        ("FOREST CITY DRY CLEANERS", 12, 30), ("COIN LAUNDRY", 3, 10),
    ],
    "Travel": [
        ("AIR CANADA", 180, 600), ("WESTJET", 150, 500), ("PORTER AIRLINES", 140, 400),
        ("VIA RAIL CANADA", 40, 140), ("AIRBNB", 120, 500), ("EXPEDIA", 150, 600), ("FLIXBUS", 25, 70),
        ("FLAIR AIRLINES", 90, 300),
    ],
    "Education": [
        ("WESTERN BOOK STORE", 30, 220), ("PEARSON EDUCATION", 60, 140), ("CHEGG", 15, 20),
        ("COURSERA", 49, 79), ("UDEMY", 15, 30),
    ],
}

# Expected events per month for variable categories.
RATES = {
    "Groceries": 6, "Dining": 8, "Coffee": 11, "Transport": 6, "Shopping": 3, "Entertainment": 2,
    "Health": 1.2, "Fees": 0.8, "Other": 1.2, "Travel": 0.25, "Education": 0.4,
}

PREFIXES = ["", "", "", "POS PURCHASE ", "VISA DEBIT PURCHASE ", "CONTACTLESS ", "SQ *", "TST* ", "IDP PURCHASE "]
CITIES = ["LONDON ON", "LONDON ON", "LONDON ON", "TORONTO ON", "KITCHENER ON", "ST THOMAS ON"]
NO_DECORATION = {"Fees", "Income", "Transfer", "Rent"}


def _decorate(merchant: str, category: str, rng: np.random.Generator, store: int) -> str:
    if category in NO_DECORATION:
        return merchant
    parts = []
    prefix = rng.choice(PREFIXES)
    if prefix in ("SQ *", "TST* ") and category not in ("Dining", "Coffee", "Groceries"):
        prefix = ""
    parts.append(f"{prefix}{merchant}")
    if rng.random() < 0.6:
        parts.append(f"#{store}")
    if rng.random() < 0.15:
        parts.append(f"{int(rng.integers(1, 13)):02d}/{int(rng.integers(1, 29)):02d}")
    if rng.random() < 0.7 and "AMAZON" not in merchant and "UBER" not in merchant:
        parts.append(rng.choice(CITIES))
    return " ".join(parts)


def generate(months: int = 12, seed: int = 7, end: date | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = end or date.today().replace(day=1) - timedelta(days=1)
    start = (pd.Timestamp(end) - pd.DateOffset(months=months) + pd.Timedelta(days=1)).date()
    stores = {}  # merchant -> a couple of store numbers so normalized names repeat
    rows: list[tuple[date, str, float, str]] = []

    def add(d, merchant, amount, category, decorate=True):
        if merchant not in stores:
            stores[merchant] = rng.integers(100, 9999, size=2)
        desc = _decorate(merchant, category, rng, int(rng.choice(stores[merchant]))) if decorate else merchant
        rows.append((d, desc, round(float(amount), 2), category))

    # Each simulated person only shops at a subset of merchants, which is realistic and
    # leaves the rest of the pool for unseen-merchant evaluation.
    favourites = {
        cat: [i for i in range(len(pool)) if rng.random() < 0.7] or [0] for cat, pool in VARIABLE.items()
    }

    month_starts = pd.date_range(start, end, freq="MS")
    for ms in month_starts:
        ms = ms.date()
        dim = pd.Period(ms, "M").days_in_month
        day = lambda d: ms.replace(day=min(d, dim))  # noqa: E731

        # Fixed monthly obligations
        add(day(1), "PREAUTHORIZED DEBIT NORTHSIDE PROPERTY MGMT", -875.00, "Rent", False)
        add(day(12), "LONDON HYDRO", -rng.uniform(45, 95), "Utilities", False)
        add(day(18), "ENBRIDGE GAS", -rng.uniform(25, 130), "Utilities", False)
        add(day(5), "ROGERS WIRELESS", -55.37, "Phone & Internet", False)
        add(day(22), "START.CA INTERNET", -62.15, "Phone & Internet", False)
        for name, amt, d in [
            ("SPOTIFY P1A2B3C4", -12.42, 3), ("NETFLIX.COM", -18.63, 9), ("CRUNCHYROLL", -9.03, 14),
            ("APPLE.COM/BILL", -3.94, 20), ("GOODLIFE FITNESS CLUBS", -45.18, 15),
        ]:
            add(day(d), name, amt, "Health" if "GOODLIFE" in name else "Subscriptions", False)
        add(day(28), "TRANSFER TO SAVINGS ACCOUNT", -150.00, "Transfer", False)
        add(day(25), "PAYMENT - VISA CARD", -rng.uniform(150, 400), "Transfer", False)

        # Income: biweekly pay, GST credit quarterly, occasional e-transfer in
        for pay_day in (8, 22):
            add(day(pay_day), "PAYROLL DEP FOREST CITY HOSPITALITY", rng.uniform(640, 910), "Income", False)
        if ms.month in (1, 4, 7, 10):
            add(day(5), "GST CANADA FED", 129.75, "Income", False)
        if rng.random() < 0.5:
            add(day(int(rng.integers(1, 28))), "INTERAC E-TRANSFER RECEIVED", rng.uniform(20, 120), "Income", False)
        if ms.month in (9, 1):
            add(day(6), "OSAP DEPOSIT ONTARIO", rng.uniform(2400, 3600), "Income", False)
            add(day(10), "WESTERN UNIVERSITY TUITION", -rng.uniform(3400, 3900), "Education", False)
        if ms.month == 11:
            add(day(21), "AMAZON PRIME MEMBERSHIP", -99.00, "Subscriptions", False)

        # Variable spending
        for cat, rate in RATES.items():
            for _ in range(rng.poisson(rate)):
                pool = VARIABLE[cat]
                i = int(rng.choice(favourites[cat])) if rng.random() < 0.85 else int(rng.integers(len(pool)))
                merchant, low, high = pool[i]
                add(day(int(rng.integers(1, dim + 1))), merchant, -rng.uniform(low, high), cat)

    df = pd.DataFrame(rows, columns=["date", "description", "amount", "category"])
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
    df["account"] = "Chequing"
    return df.sort_values("date").reset_index(drop=True)


def write_rbc_csv(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Account Type", "Account Number", "Transaction Date", "Cheque Number",
                    "Description 1", "Description 2", "CAD$", "USD$"])
        for r in df.itertuples():
            w.writerow(["Chequing", "01234-5678901", r.date.strftime("%m/%d/%Y"), "",
                        r.description, "", f"{r.amount:.2f}", ""])


def write_td_csv(df: pd.DataFrame, path: Path) -> None:
    """TD: no header; date, description, debit, credit, balance."""
    bal = 2500.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for r in df.itertuples():
            bal += r.amount
            debit = f"{-r.amount:.2f}" if r.amount < 0 else ""
            credit = f"{r.amount:.2f}" if r.amount > 0 else ""
            w.writerow([r.date.strftime("%m/%d/%Y"), r.description, debit, credit, f"{bal:.2f}"])


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "sample_data"
    out.mkdir(exist_ok=True)
    data = generate()
    data.to_csv(out / "labelled_demo.csv", index=False)
    write_rbc_csv(data, out / "rbc_export_example.csv")
    write_td_csv(data, out / "td_export_example.csv")
    print(f"{len(data)} transactions, {data['category'].nunique()} categories -> {out}")
