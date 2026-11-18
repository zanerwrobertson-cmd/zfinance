"""App workflows that touch the store, kept out of the UI so they can be tested."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .categorizer import Categorizer, evaluate
from .db import Store
from .synth import generate


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
    cutoff = demo["date"].min() + pd.DateOffset(months=2)
    demo.loc[demo["date"] >= cutoff, "category"] = None
    inserted, skipped = store.add_transactions(demo)
    msg = f"Loaded {inserted} demo transactions ({skipped} duplicates skipped)."
    return f"{msg} {retrain(store)}" if inserted else msg
