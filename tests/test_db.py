import pandas as pd

from zfinance.db import Store
from zfinance.importers import parse_csv

TD = """10/01/2025,MCDONALD'S #7326,10.06,,1614.94
10/02/2025,SECOND CUP,2.10,,1612.84
10/02/2025,SECOND CUP,2.10,,1610.74
"""


def test_reimport_is_deduplicated(tmp_path):
    store = Store(tmp_path / "t.db")
    df = parse_csv(TD).df
    assert store.add_transactions(df) == (3, 0)  # same-day duplicate coffees both kept
    assert store.add_transactions(df) == (0, 3)
    assert len(store.transactions()) == 3


def test_overlapping_export_only_adds_new_rows(tmp_path):
    store = Store(tmp_path / "t.db")
    store.add_transactions(parse_csv(TD).df)
    newer = TD + "10/03/2025,METRO,40.00,,1570.74\n"
    assert store.add_transactions(parse_csv(newer).df) == (1, 3)


def test_predictions_never_overwrite_user_labels(tmp_path):
    store = Store(tmp_path / "t.db")
    store.add_transactions(parse_csv(TD).df)
    ids = store.transactions()["id"].tolist()
    store.set_labels({ids[0]: "Dining"})
    store.set_predictions(ids, ["Dining"] * 3, [0.5] * 3)
    t = store.transactions().set_index("id")
    assert t.loc[ids[0], "category"] == "Dining"
    assert t.loc[ids[0], "label_source"] == "user"
    assert (t.loc[ids[1:], "label_source"] == "model").all()
