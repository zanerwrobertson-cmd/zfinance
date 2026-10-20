"""Benchmark the categorizer and write reports/evaluation.md.

Measures, averaged over several simulated users (different random seeds):
  * majority-class and keyword-rule baselines
  * model accuracy on a random split vs. on merchants never seen in training
  * the realistic scenario: label your first N months, how well does it handle the rest?
    Reported as accuracy, plus automation rate and precision at a 0.8 confidence threshold
    (transactions above the threshold are auto-accepted, the rest go to review).

Usage:
    python scripts/evaluate.py            # synthetic data, 5 simulated users
    python scripts/evaluate.py --db data/zfinance.db   # your own labelled transactions
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from zfinance.baseline import rule_categorize  # noqa: E402
from zfinance.categorizer import Categorizer, evaluate  # noqa: E402
from zfinance.db import Store  # noqa: E402
from zfinance.normalize import normalize_description  # noqa: E402
from zfinance.synth import generate  # noqa: E402

THRESHOLD = 0.8


def temporal(df: pd.DataFrame, train_months: int) -> dict:
    months = sorted(df["date"].dt.to_period("M").unique())
    if len(months) <= train_months:
        return {}
    cutoff = months[train_months]
    train = df[df["date"].dt.to_period("M") < cutoff]
    test = df[df["date"].dt.to_period("M") >= cutoff]
    model = Categorizer().fit(train)
    preds, confs = model.predict(test)
    preds, confs = np.array(preds), np.array(confs)
    y = test["category"].to_numpy()
    auto = confs >= THRESHOLD
    return {
        "train_txns": len(train),
        "accuracy": float((preds == y).mean()),
        "automation_rate": float(auto.mean()),
        "precision_auto": float((preds[auto] == y[auto]).mean()) if auto.any() else float("nan"),
        "rule_accuracy": float((test["clean_desc"].map(rule_categorize).to_numpy() == y).mean()),
    }


def run(datasets: list[pd.DataFrame]) -> str:
    static, temp = [], {k: [] for k in (1, 2, 3, 6)}
    for df in datasets:
        df = df.copy()
        df["clean_desc"] = df["description"].map(normalize_description)
        df = df.reset_index(drop=True)
        df["id"] = df.index
        rep = evaluate(df)
        static.append(
            {
                "n": len(df),
                "merchants": rep.n_merchants,
                "majority": df["category"].value_counts(normalize=True).iloc[0],
                "rules": (df["clean_desc"].map(rule_categorize) == df["category"]).mean(),
                "random": rep.random_split_accuracy,
                "unseen": rep.unseen_merchant_accuracy,
                "unseen_f1": rep.unseen_merchant_macro_f1,
            }
        )
        for k in temp:
            r = temporal(df, k)
            if r:
                temp[k].append(r)

    s = pd.DataFrame(static).mean()
    pct = lambda v: f"{v * 100:.1f}%"  # noqa: E731
    lines = [
        "# zfinance evaluation",
        "",
        f"Averaged over {len(datasets)} simulated users "
        f"(~{s['n']:.0f} transactions, ~{s['merchants']:.0f} distinct merchant strings each).",
        "",
        "## Overall accuracy (5-fold cross-validation)",
        "",
        "| Method | Accuracy |",
        "|---|---|",
        f"| Majority class | {pct(s['majority'])} |",
        f"| Keyword rules | {pct(s['rules'])} |",
        f"| Model, random split | {pct(s['random'])} |",
        f"| Model, unseen merchants only | {pct(s['unseen'])} (macro-F1 {s['unseen_f1']:.2f}) |",
        "",
        "The random split lets the same merchant appear in train and test, which flatters the "
        "model. The unseen-merchant score (GroupKFold by normalized merchant) is the honest "
        "measure of generalization.",
        "",
        "## Realistic use: label the first N months, predict the rest",
        "",
        f"Auto-accept threshold: confidence >= {THRESHOLD}.",
        "",
        "| Months labelled | Model accuracy | Keyword rules | Auto-categorized | Precision on auto-categorized |",
        "|---|---|---|---|---|",
    ]
    for k, rows in temp.items():
        if not rows:
            continue
        t = pd.DataFrame(rows).mean()
        lines.append(
            f"| {k} | {pct(t['accuracy'])} | {pct(t['rule_accuracy'])} | "
            f"{pct(t['automation_rate'])} | {pct(t['precision_auto'])} |"
        )
    lines += [
        "",
        "Caveat: the data is synthetic and the keyword rules were written with knowledge of "
        "the merchant list, which favours the baseline. Re-run with `--db` on your own "
        "labelled data for real numbers.",
    ]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="Evaluate on user-labelled transactions in this SQLite DB")
    ap.add_argument("--users", type=int, default=5)
    args = ap.parse_args()

    if args.db:
        df = Store(args.db).transactions()
        df = df[df["label_source"] == "user"]
        datasets = [df]
    else:
        datasets = [generate(seed=s) for s in range(args.users)]

    report = run(datasets)
    out = ROOT / "reports" / "evaluation.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
