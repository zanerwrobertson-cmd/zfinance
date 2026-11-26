"""Evaluate the categorizer on your own labelled transactions. Prints aggregates only.

Merchant-grouped cross-validation (no merchant is in both train and test) comparing the
model with merchant memory against the keyword-rule and majority-class baselines.
Never prints individual transactions, merchants or amounts.

Usage:
    python scripts/eval_real.py [--db data/zfinance.db]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluate import THRESHOLD  # noqa: E402  (scripts/ is on sys.path when run directly)
from zfinance.baseline import rule_categorize  # noqa: E402
from zfinance.categorizer import Categorizer  # noqa: E402
from zfinance.db import Store  # noqa: E402
from zfinance.normalize import merchant_key  # noqa: E402

# Below these, fold-to-fold variance swamps the differences between methods.
MIN_RELIABLE_LABELS = 200
MIN_RELIABLE_MERCHANTS = 30
MIN_RELIABLE_AUTO = 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "zfinance.db"))
    args = ap.parse_args()

    df = Store(args.db).transactions()
    df = df[df["label_source"] == "user"].reset_index(drop=True)
    ok, msg = Categorizer.can_train(df)
    if not ok:
        print(f"Too few labels to evaluate. {msg}")
        return

    y = df["category"].astype(str).to_numpy()
    groups = df["clean_desc"].map(merchant_key).to_numpy()
    n_groups = len(set(groups))
    model_p = np.empty(len(df), dtype=object)
    conf = np.zeros(len(df))
    major_p = np.empty(len(df), dtype=object)
    for tr, te in GroupKFold(n_splits=max(2, min(5, n_groups))).split(df, y, groups):
        train, test = df.iloc[tr], df.iloc[te]
        major_p[te] = train["category"].mode().iloc[0]
        try:
            preds, confs = Categorizer().fit(train).predict(test)
            model_p[te], conf[te] = preds, confs
        except ValueError:  # training fold too small or single-class: fall back to majority, never auto-accept
            model_p[te], conf[te] = major_p[te], 0.0
    rule_p = df["clean_desc"].map(rule_categorize).to_numpy()

    periods = df["date"].dt.to_period("M")
    months = (periods.max() - periods.min()).n + 1
    auto = conf >= THRESHOLD
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731

    print(f"Labelled transactions: {len(df)}")
    print(f"Date range:            {months} month(s)")
    print(f"Categories:            {df['category'].nunique()}")
    print(f"Distinct merchants:    {n_groups}")
    print("\nAccuracy (merchant-grouped CV, test merchants unseen in training)")
    print(f"  Majority class:        {pct((major_p == y).mean())}")
    print(f"  Keyword rules:         {pct((rule_p == y).mean())}")
    print(f"  Model + memory:        {pct((model_p == y).mean())}")
    print(f"\nAt auto-accept threshold {THRESHOLD}:")
    print(f"  Coverage (auto-accepted): {pct(auto.mean())}")
    print(f"  Precision:                {pct((model_p[auto] == y[auto]).mean()) if auto.any() else 'n/a (none above threshold)'}")

    reasons = []
    if len(df) < MIN_RELIABLE_LABELS:
        reasons.append(f"fewer than {MIN_RELIABLE_LABELS} labels")
    if n_groups < MIN_RELIABLE_MERCHANTS:
        reasons.append(f"fewer than {MIN_RELIABLE_MERCHANTS} distinct merchants")
    if auto.sum() < MIN_RELIABLE_AUTO:
        reasons.append(f"fewer than {MIN_RELIABLE_AUTO} auto-accepted predictions (precision is noisy)")
    if reasons:
        print("\nWARNING: too few labels for a reliable estimate: " + "; ".join(reasons) + ".")


if __name__ == "__main__":
    main()
