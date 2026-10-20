"""Summaries computed from categorized transactions."""

from __future__ import annotations

import numpy as np
import pandas as pd

NON_SPENDING = {"Income", "Transfer"}


def _spending(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["amount"] < 0) & (~df["category"].isin(NON_SPENDING))].copy()
    d["spend"] = -d["amount"]
    return d


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Income, spending, net and savings rate per month."""
    if df.empty:
        return pd.DataFrame(columns=["month", "income", "spending", "net", "savings_rate"])
    d = df.copy()
    d["month"] = d["date"].dt.to_period("M")
    income = d[(d["amount"] > 0) & (d["category"] != "Transfer")].groupby("month")["amount"].sum()
    spend = _spending(d).groupby("month")["spend"].sum()
    out = pd.DataFrame({"income": income, "spending": spend}).fillna(0.0)
    out["net"] = out["income"] - out["spending"]
    out["savings_rate"] = np.where(out["income"] > 0, out["net"] / out["income"], np.nan)
    out = out.sort_index().reset_index().rename(columns={"index": "month"})
    out["month"] = out["month"].astype(str)
    return out


def category_by_month(df: pd.DataFrame) -> pd.DataFrame:
    d = _spending(df)
    if d.empty:
        return pd.DataFrame(columns=["month", "category", "spend"])
    d["month"] = d["date"].dt.to_period("M").astype(str)
    return d.groupby(["month", "category"], as_index=False)["spend"].sum()


def budget_status(df: pd.DataFrame, budgets: dict[str, float], month: str) -> pd.DataFrame:
    """Spend vs limit for one month, with a straight-line projection to month end."""
    d = _spending(df)
    d = d[d["date"].dt.to_period("M").astype(str) == month]
    spent = d.groupby("category")["spend"].sum()

    period = pd.Period(month, "M")
    days_in_month = period.days_in_month
    last_day = d["date"].max().day if not d.empty else days_in_month
    today = pd.Timestamp.today()
    if period == today.to_period("M"):
        last_day = today.day
    frac = max(last_day, 1) / days_in_month

    rows = []
    for cat, limit in sorted(budgets.items()):
        s = float(spent.get(cat, 0.0))
        projected = s / frac if frac < 1 else s
        rows.append(
            {
                "category": cat,
                "limit": limit,
                "spent": s,
                "remaining": limit - s,
                "pct_used": s / limit if limit else np.nan,
                "projected": projected,
                "on_track": projected <= limit,
            }
        )
    return pd.DataFrame(rows)


def detect_recurring(df: pd.DataFrame, min_occurrences: int = 3) -> pd.DataFrame:
    """Find charges that repeat at a regular interval with a stable amount.

    A merchant counts as recurring if it has >= min_occurrences charges, the median gap
    between them is roughly weekly, monthly or yearly, gaps are consistent, and amounts
    vary by under 15%.
    """
    d = df[df["amount"] < 0].copy()
    rows = []
    for merchant, g in d.groupby("clean_desc"):
        if len(g) < min_occurrences:
            continue
        g = g.sort_values("date")
        gaps = g["date"].diff().dt.days.dropna()
        med_gap = gaps.median()
        if med_gap <= 0:
            continue
        if 5 <= med_gap <= 9:
            cadence, per_year = "Weekly", 52
        elif 26 <= med_gap <= 35:
            cadence, per_year = "Monthly", 12
        elif 350 <= med_gap <= 380:
            cadence, per_year = "Yearly", 1
        else:
            continue
        if (gaps - med_gap).abs().median() > 0.25 * med_gap:
            continue
        amounts = -g["amount"]
        if amounts.std(ddof=0) / amounts.mean() > 0.15:
            continue
        rows.append(
            {
                "merchant": g["description"].iloc[-1],
                "category": g["category"].iloc[-1],
                "cadence": cadence,
                "typical_amount": round(float(amounts.median()), 2),
                "annual_cost": round(float(amounts.median()) * per_year, 2),
                "last_charged": g["date"].iloc[-1].date(),
                "occurrences": len(g),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values("annual_cost", ascending=False).reset_index(drop=True) if not out.empty else out
