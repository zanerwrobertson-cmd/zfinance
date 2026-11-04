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


def page_review():
    st.header("Review")
    df = store.transactions()
    if df.empty:
        st.info("Import some transactions first.")
        return
    queue = df[df["label_source"] != "user"].sort_values("confidence", na_position="first")
    c1, c2, c3 = st.columns(3)
    c1.metric("To review", len(queue))
    c2.metric("User labelled", int((df["label_source"] == "user").sum()))
    runs = store.model_runs()
    if not runs.empty and pd.notna(runs["cv_macro_f1"].iloc[-1]):
        c3.metric("Unseen-merchant macro-F1", f"{runs['cv_macro_f1'].iloc[-1]:.2f}")

    if c1.button("Retrain model"):
        st.info(retrain())
        st.rerun()
    if queue.empty:
        st.success("Nothing to review.")
        return

    n = st.slider("Rows to show", 10, 200, 50)
    shown = queue.head(n)[["id", "date", "description", "amount", "category", "confidence"]]
    select_all = st.checkbox("Select all", help="Tick confirm on every row shown")
    shown = shown.assign(confirm=select_all)
    edited = st.data_editor(
        shown,
        hide_index=True,
        width="stretch",
        disabled=["id", "date", "description", "amount", "confidence"],
        column_config={
            "category": st.column_config.SelectboxColumn(options=CATEGORIES),
            "confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
            "confirm": st.column_config.CheckboxColumn(help="Accept this category as a label"),
        },
    )
    same_merchant = st.checkbox("Apply each label to every transaction from the same merchant", True)

    if st.button("Save labels", type="primary"):
        changed = edited[(edited["category"] != shown["category"].values) | edited["confirm"]]
        changed = changed[changed["category"].notna()]
        labels = dict(zip(changed["id"], changed["category"]))
        if same_merchant:
            clean = df.set_index("id")["clean_desc"]
            for i, cat in list(labels.items()):
                for j in df.loc[df["clean_desc"] == clean[i], "id"]:
                    labels[j] = cat
        store.set_labels(labels)
        st.success(f"Saved {len(labels)} labels. Retrain to update predictions.")


def page_dashboard():
    st.header("Dashboard")
    df = store.transactions()
    df = df[df["category"].notna()]
    if df.empty:
        st.info("No categorized transactions yet.")
        return

    ms = monthly_summary(df)
    last = ms.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Income ({last['month']})", f"${last['income']:,.0f}")
    c2.metric("Spending", f"${last['spending']:,.0f}")
    c3.metric("Net", f"${last['net']:,.0f}")
    c4.metric("Savings rate", "—" if pd.isna(last["savings_rate"]) else f"{last['savings_rate']:.0%}")

    long = ms.melt("month", ["income", "spending"], "type", "value")
    st.plotly_chart(px.bar(long, x="month", y="value", color="type", barmode="group",
                           title="Income vs spending"), width="stretch")
    st.plotly_chart(px.line(ms, x="month", y="savings_rate", markers=True, title="Savings rate")
                    .update_yaxes(tickformat=".0%"), width="stretch")

    cbm = category_by_month(df)
    left, right = st.columns(2)
    left.plotly_chart(px.bar(cbm, x="month", y="spend", color="category", title="Spending by category"),
                      width="stretch")
    month = right.selectbox("Month", sorted(cbm["month"].unique(), reverse=True))
    right.plotly_chart(px.pie(cbm[cbm["month"] == month], names="category", values="spend", hole=0.4),
                       width="stretch")

    st.subheader("Recurring charges")
    rec = detect_recurring(df)
    if rec.empty:
        st.caption("None detected.")
    else:
        st.metric("Total per year", f"${rec['annual_cost'].sum():,.0f}")
        st.dataframe(rec, hide_index=True, width="stretch")



pages = {"Import": page_import, "Review": page_review, "Dashboard": page_dashboard}
choice = st.sidebar.radio("zfinance", list(pages))
pages[choice]()
