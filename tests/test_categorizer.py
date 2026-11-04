import pandas as pd
import pytest

from zfinance.categorizer import Categorizer, evaluate
from zfinance.normalize import normalize_description
from zfinance.synth import generate


@pytest.fixture(scope="module")
def labeled():
    df = generate(months=6, seed=1)
    df["clean_desc"] = df["description"].map(normalize_description)
    df["id"] = range(len(df))
    return df


def test_can_train_requires_enough_labels(labeled):
    ok, _ = Categorizer.can_train(labeled.head(5))
    assert not ok
    one_class = labeled[labeled["category"] == labeled["category"].iloc[0]].head(20)
    ok, _ = Categorizer.can_train(one_class)
    assert not ok
    with pytest.raises(ValueError):
        Categorizer().fit(labeled.head(5))


def test_fit_predict_on_seen_merchants(labeled):
    model = Categorizer().fit(labeled)
    cats, confs = model.predict(labeled)
    assert len(cats) == len(labeled)
    assert all(0 <= c <= 1 for c in confs)
    assert (pd.Series(cats) == labeled["category"].values).mean() > 0.95


def test_merchant_memory_uses_latest_label(labeled):
    df = labeled.copy()
    target = df["clean_desc"].iloc[0]
    df.loc[df.index[-1], ["clean_desc", "category"]] = [target, "Other"]
    model = Categorizer().fit(df)
    cats, confs = model.predict(df[df["clean_desc"] == target])
    assert set(cats) == {"Other"}
    assert min(confs) >= 0.99


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        Categorizer().predict(pd.DataFrame({"clean_desc": ["x"], "amount": [-1.0]}))


def test_evaluate_reports_leakage_gap(labeled):
    # The random split needs >= 2 examples per class; rare synthetic classes vary by seed.
    counts = labeled["category"].value_counts()
    common = labeled[labeled["category"].isin(counts[counts >= 3].index)]
    rep = evaluate(common, n_splits=3)
    assert rep.n_labels == len(common)
    # Random split leaks merchants across folds, so it should beat unseen-merchant accuracy.
    assert rep.random_split_accuracy > rep.unseen_merchant_accuracy
