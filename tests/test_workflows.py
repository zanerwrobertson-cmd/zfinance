from datetime import date

import pytest

from zfinance.analytics import budget_status
from zfinance.db import Store
from zfinance.synth import DEFAULT_END, generate
from zfinance.workflows import DEMO_BUDGET_CATEGORIES, demo_budgets, seed_demo


@pytest.fixture(scope="module")
def seeded(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("demo") / "t.db")
    seed_demo(store)
    return store


def test_seed_demo_labels_predicts_and_budgets(seeded):
    sources = seeded.transactions()["label_source"].value_counts()
    assert sources["user"] > 0 and sources["model"] > 0
    assert set(seeded.budgets()) == set(DEMO_BUDGET_CATEGORIES)


def test_seed_demo_twice_adds_no_rows(seeded):
    before = len(seeded.transactions())
    assert seed_demo(seeded).startswith("Loaded 0 demo transactions")
    assert len(seeded.transactions()) == before


def test_seed_demo_ends_today_but_benchmark_data_does_not(seeded):
    latest = seeded.transactions()["date"].max()
    assert latest.date() <= date.today()
    assert latest.strftime("%Y-%m") == date.today().strftime("%Y-%m")
    assert DEFAULT_END == date(2026, 9, 30)
    assert generate()["date"].max().strftime("%Y-%m") == "2026-09"


def test_demo_budgets_include_one_over_pace():
    demo = generate(end=date(2026, 9, 15))
    status = budget_status(demo, demo_budgets(demo), "2026-09")
    assert (~status["on_track"]).sum() == 1
