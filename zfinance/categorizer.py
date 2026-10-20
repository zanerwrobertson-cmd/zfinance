"""Transaction categorizer.

Two layers:

1. Merchant memory: if you've already labelled this exact (normalized) merchant, reuse
   your most recent label. Cheap, and correct ~always once a merchant has been seen.
2. ML model for everything else: TF-IDF over word n-grams + character n-grams of the
   cleaned description, plus a few amount features, into a multinomial logistic
   regression. Character n-grams are what let it generalize to merchants it has never
   seen ("golden bakery" -> Dining because "bakery"; "shell canada" -> Transport).

Predictions carry a probability. Low-confidence ones go to the review queue, you
correct them, the model retrains. That loop is the point of the app.

Evaluation uses GroupKFold grouped by merchant, so the score measures performance on
merchants absent from training. A plain stratified split leaks the same merchant into
train and test and inflates accuracy; both numbers are reported so the gap is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .normalize import merchant_key

MIN_LABELS = 10
MIN_CLASSES = 2


class AmountFeatures(BaseEstimator, TransformerMixin):
    """sign, log|amount|, whole-dollar flag (rent/transfers are usually round)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=float).reshape(-1)
        return np.column_stack(
            [
                (a > 0).astype(float),
                np.log1p(np.abs(a)),
                (np.abs(a - np.round(a)) < 1e-9).astype(float),
            ]
        )


def build_pipeline() -> Pipeline:
    features = ColumnTransformer(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), "clean_desc"),
            (
                "char",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True),
                "clean_desc",
            ),
            ("amount", Pipeline([("f", AmountFeatures()), ("s", StandardScaler())]), "amount"),
        ]
    )
    clf = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


@dataclass
class EvalReport:
    n_labels: int
    n_merchants: int
    unseen_merchant_accuracy: float
    unseen_merchant_macro_f1: float
    random_split_accuracy: float
    per_class_f1: dict[str, float] = field(default_factory=dict)


@dataclass
class Categorizer:
    pipeline: Pipeline | None = None
    merchant_memory: dict[str, str] = field(default_factory=dict)
    classes_: list[str] = field(default_factory=list)

    @staticmethod
    def can_train(labeled: pd.DataFrame) -> tuple[bool, str]:
        if len(labeled) < MIN_LABELS:
            return False, f"Need at least {MIN_LABELS} labelled transactions (have {len(labeled)})."
        if labeled["category"].nunique() < MIN_CLASSES:
            return False, f"Need at least {MIN_CLASSES} different categories labelled."
        return True, ""

    def fit(self, labeled: pd.DataFrame) -> "Categorizer":
        ok, msg = self.can_train(labeled)
        if not ok:
            raise ValueError(msg)
        X = labeled[["clean_desc", "amount"]]
        y = labeled["category"].astype(str)
        self.pipeline = build_pipeline().fit(X, y)
        self.classes_ = list(self.pipeline.classes_)
        # Most recent label per merchant wins (labeled is assumed newest-first or any order;
        # sort by id so later corrections override earlier ones).
        ordered = labeled.sort_values("id") if "id" in labeled.columns else labeled
        self.merchant_memory = dict(zip(ordered["clean_desc"], ordered["category"]))
        return self

    def predict(self, df: pd.DataFrame) -> tuple[list[str], list[float]]:
        if self.pipeline is None:
            raise RuntimeError("Model is not trained.")
        if df.empty:
            return [], []
        proba = self.pipeline.predict_proba(df[["clean_desc", "amount"]])
        idx = proba.argmax(axis=1)
        cats = [self.classes_[i] for i in idx]
        confs = proba[np.arange(len(idx)), idx].tolist()
        for k, desc in enumerate(df["clean_desc"]):
            if desc in self.merchant_memory:
                cats[k] = self.merchant_memory[desc]
                confs[k] = max(confs[k], 0.99)
        return cats, confs

    def top_terms(self, category: str, n: int = 10) -> list[str]:
        """Most positive word features for a class — shows what the model learned."""
        if self.pipeline is None or category not in self.classes_:
            return []
        feats = self.pipeline.named_steps["features"]
        clf = self.pipeline.named_steps["clf"]
        word_vec = feats.named_transformers_["word"]
        vocab = word_vec.get_feature_names_out()
        coefs = clf.coef_[self.classes_.index(category)][: len(vocab)]
        return [vocab[i] for i in np.argsort(coefs)[::-1][:n]]


def evaluate(labeled: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> EvalReport:
    """Cross-validated scores. Grouped-by-merchant is the honest number."""
    ok, msg = Categorizer.can_train(labeled)
    if not ok:
        raise ValueError(msg)
    labeled = labeled.reset_index(drop=True)
    X = labeled[["clean_desc", "amount"]]
    y = labeled["category"].astype(str).to_numpy()
    groups = labeled["clean_desc"].map(merchant_key).to_numpy()
    n_groups = len(set(groups))

    # Unseen-merchant evaluation: no merchant appears in both train and test.
    k = max(2, min(n_splits, n_groups))
    preds = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(set(y[tr])) < 2:
            preds[te] = y[tr][0]
            continue
        model = build_pipeline().fit(X.iloc[tr], y[tr])
        preds[te] = model.predict(X.iloc[te])
    acc_g = accuracy_score(y, preds)
    f1_g = f1_score(y, preds, average="macro", zero_division=0)
    labels = sorted(set(y))
    per_class = dict(zip(labels, f1_score(y, preds, labels=labels, average=None, zero_division=0)))

    # Random (stratified) split for comparison. Same merchants leak across folds.
    min_class = pd.Series(y).value_counts().min()
    k2 = max(2, min(n_splits, int(min_class)))
    preds_r = np.empty(len(y), dtype=object)
    if min_class >= 2:
        for tr, te in StratifiedKFold(n_splits=k2, shuffle=True, random_state=seed).split(X, y):
            preds_r[te] = build_pipeline().fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
        acc_r = accuracy_score(y, preds_r)
    else:
        acc_r = float("nan")

    return EvalReport(
        n_labels=len(y),
        n_merchants=n_groups,
        unseen_merchant_accuracy=float(acc_g),
        unseen_merchant_macro_f1=float(f1_g),
        random_split_accuracy=float(acc_r),
        per_class_f1={k: float(v) for k, v in per_class.items()},
    )
