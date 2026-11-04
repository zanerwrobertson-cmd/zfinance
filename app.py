"""zfinance Streamlit app. Run with: streamlit run app.py"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from zfinance import CATEGORIES
from zfinance.analytics import budget_status, category_by_month, detect_recurring, monthly_summary
from zfinance.categorizer import Categorizer, evaluate
from zfinance.db import Store
from zfinance.importers import ImportFormatError, parse_csv, parse_mapped, preview_rows
from zfinance.synth import generate

st.set_page_config(page_title="zfinance", layout="wide")
store = Store()


def retrain() -> str:
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


def import_parsed(df: pd.DataFrame) -> None:
    inserted, skipped = store.add_transactions(df)
    st.success(f"Imported {inserted} transactions ({skipped} duplicates skipped).")
    if inserted:
        st.info(retrain())


# ---------------- pages ----------------


def page_import():
    st.header("Import")
    up = st.file_uploader("Bank CSV export", type=["csv"])
    account = st.text_input("Account name", "Chequing")
    invert = st.toggle("Invert sign (use if spending shows up as positive)")

    if up is not None:
        data = up.getvalue()
        try:
            res = parse_csv(data, account_name=account, invert_sign=invert)
            st.caption(f"Detected format: {res.source_format}")
            for w in res.warnings:
                st.warning(w)
            st.dataframe(res.df.head(20), width="stretch")
            if st.button("Import these transactions", type="primary"):
                import_parsed(res.df)
        except ImportFormatError as e:
            st.error(f"{e}")
            rows = preview_rows(data)
            st.dataframe(pd.DataFrame(rows), width="stretch")
            width = max(len(r) for r in rows)
            cols = list(range(width))
            c1, c2, c3, c4 = st.columns(4)
            date_col = c1.selectbox("Date column", cols, 0)
            desc_col = c2.selectbox("Description column", cols, min(1, width - 1))
            amt_col = c3.selectbox("Amount column", cols, min(2, width - 1))
            skip = c4.number_input("Skip rows", 0, 20, 0)
            if st.button("Import with this mapping", type="primary"):
                res = parse_mapped(data, date_col, desc_col, amt_col, int(skip), account, invert)
                if res.df.empty:
                    st.error("No rows parsed with that mapping.")
                else:
                    import_parsed(res.df)

    st.divider()
    st.subheader("Demo data")
    st.caption("12 months of synthetic transactions. The first 2 months keep their labels; the model categorizes the rest.")
    c1, c2 = st.columns(2)
    if c1.button("Load demo data"):
        demo = generate()
        cutoff = demo["date"].min() + pd.DateOffset(months=2)
        demo.loc[demo["date"] >= cutoff, "category"] = None
        import_parsed(demo)
    if c2.button("Delete all transactions"):
        store.clear()
        st.success("Cleared.")



pages = {"Import": page_import}
choice = st.sidebar.radio("zfinance", list(pages))
pages[choice]()
