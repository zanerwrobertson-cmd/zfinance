"""Keyword-rule baseline: the approach most budgeting apps start with.

Used only in evaluation, so the ML model has something honest to beat.
"""

from __future__ import annotations

RULES: list[tuple[str, list[str]]] = [
    ("Income", ["payroll", "pay dep", "deposit", "gst", "e-transfer received", "osap"]),
    ("Transfer", ["transfer to", "payment - visa", "savings"]),
    ("Rent", ["rent", "property", "landlord"]),
    ("Utilities", ["hydro", "gas canada", "enbridge", "water"]),
    ("Phone & Internet", ["rogers", "bell", "fido", "koodo", "freedom", "telus", "internet", "wireless"]),
    ("Subscriptions", ["spotify", "netflix", "disney", "crave", "prime", "apple.com", "youtube"]),
    ("Coffee", ["tim hortons", "starbucks", "coffee", "cafe", "espresso"]),
    ("Dining", ["restaurant", "pizza", "burger", "uber eats", "doordash", "skip", "mcdonald", "subway", "grill", "shawarma"]),
    ("Groceries", ["grocery", "metro", "no frills", "loblaws", "sobeys", "food basics", "costco", "market"]),
    ("Transport", ["transit", "uber trip", "lyft", "petro", "esso", "shell", "gas bar", "parking"]),
    ("Health", ["pharmacy", "drug mart", "dental", "physio", "fitness", "optometry"]),
    ("Entertainment", ["cineplex", "steam", "playstation", "ticketmaster", "nintendo"]),
    ("Travel", ["air canada", "westjet", "airbnb", "expedia", "via rail", "airlines"]),
    ("Education", ["tuition", "university", "book store", "bookstore", "coursera", "chegg"]),
    ("Fees", ["fee", "service charge", "interest"]),
    ("Shopping", ["amazon", "walmart", "canadian tire", "best buy", "winners", "dollarama", "ikea"]),
]


def rule_categorize(clean_desc: str) -> str:
    for category, keywords in RULES:
        if any(k in clean_desc for k in keywords):
            return category
    return "Other"
