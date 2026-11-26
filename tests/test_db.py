import pandas as pd

import pytest

from zfinance.db import DEFAULT_DB, Store, device_db
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


def test_stores_on_different_paths_are_isolated(tmp_path):
    a, b = Store(tmp_path / "a.db"), Store(tmp_path / "b.db")
    a.add_transactions(parse_csv(TD).df)
    assert len(a.transactions()) == 3
    assert b.transactions().empty


def test_device_db_separates_by_ip():
    for local in ("127.0.0.1", "::1", "::ffff:127.0.0.1"):
        assert device_db(local) == DEFAULT_DB
    other = device_db("192.168.1.20")
    assert other != DEFAULT_DB and other.parent == DEFAULT_DB.parent / "devices"
    assert device_db("192.168.1.21") != other
    assert ":" not in device_db("fe80::1%eth0").name
    with pytest.raises(ValueError):
        device_db("../zfinance")
