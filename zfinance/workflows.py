"""App workflows that touch the store, kept out of the UI so they can be tested."""

from __future__ import annotations

import math
from datetime import date

import pandas as pd

from .analytics import budget_status, category_by_month
from .categorizer import Categorizer, evaluate
from .db import Store
from .synth import generate

DEMO_BUDGET_CATEGORIES = ("Groceries", "Dining", "Transport", "Shopping")


def retrain(store: Store) -> str:
    """Fit on user labels, predict everything not user-labelled, log CV scores."""
    df = store.transactions()
    labeled = df[df["label_source"] == "user"]
    ok, msg = Categorizer.can_train(labeled)
    if not ok:
        return msg
    model = Categorizer().fit(labeled)
    todo = df[df["label_source"] != "user"]
    cats, confs = model.predict(todo)
    store.set_predictions(todo["id"].tolist(), cats, confs)
    try:
        rep = evaluate(labeled)
        store.log_model_run(len(labeled), rep.unseen_merchant_accuracy, rep.unseen_merchant_macro_f1)
    except ValueError:
        store.log_model_run(len(labeled), None, None)
    return f"Trained on {len(labeled)} labels, categorized {len(todo)} transactions."


def seed_demo(store: Store, end: date | None = None) -> str:
    """Load the synthetic demo. The first 2 months keep their labels; the model categorizes the rest.

    The data ends today by default, so the latest month is current. The benchmark calls
    generate() directly and stays on its fixed end date.
    """
    demo = generate(end=end or date.today())
    budgets = demo_budgets(demo)  # from the true categories, before labels are hidden
    cutoff = demo["date"].min() + pd.DateOffset(months=2)
    demo.loc[demo["date"] >= cutoff, "category"] = None
    inserted, skipped = store.add_transactions(demo)
    existing = store.budgets()
    for category, limit in budgets.items():
        if category not in existing:  # never overwrite a budget someone set
            store.set_budget(category, limit)
    msg = f"Loaded {inserted} demo transactions ({skipped} duplicates skipped)."
    return f"{msg} {retrain(store)}" if inserted else msg


def demo_budgets(demo: pd.DataFrame) -> dict[str, float]:
    """Example limits from typical monthly spend, with one set tight enough to trigger a warning.

    Comfortable limits are the median month plus 15%, rounded up to $50. The tight one goes
    to the category furthest over its typical spend on this month's projection, at 80% of
    the lower of the two, rounded down to $50.
    """
    current = demo["date"].max().to_period("M").strftime("%Y-%m")
    monthly = category_by_month(demo)
    monthly = monthly[monthly["category"].isin(DEMO_BUDGET_CATEGORIES) & (monthly["month"] != current)]
    typical = monthly.groupby("category")["spend"].median()
    budgets = {c: max(50.0, math.ceil(typical[c] * 1.15 / 50) * 50) for c in typical.index}
    # shortcut: in the first day or two of a month nothing may be spent yet, so no category
    # is over and no warning shows; fine for a demo, revisit if the Budgets page needs one daily.
    status = budget_status(demo, typical.to_dict(), current).set_index("category")
    tight = (status["projected"] / status["limit"]).idxmax()
    lower = min(typical[tight], status.loc[tight, "projected"])
    budgets[tight] = max(50.0, math.floor(lower * 0.8 / 50) * 50)
    return budgets
