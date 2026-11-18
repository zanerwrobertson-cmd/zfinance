"""App workflows that touch the store, kept out of the UI so they can be tested."""

from __future__ import annotations

from .categorizer import Categorizer, evaluate
from .db import Store


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
