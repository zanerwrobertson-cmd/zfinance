# zfinance

A local-first personal finance app that imports bank CSV exports, categorizes transactions with a machine learning model, and learns from your corrections. Everything stays in a single SQLite file on your machine.

## Features

- **Import** CSV exports from Canadian banks (RBC, TD, CIBC, BMO, Scotiabank). It detects header and headerless formats automatically, and falls back to manual column mapping when it can't work out a file.
- **Deduplicated re-imports**: importing an overlapping export only adds the new rows, and genuine same-day duplicates (two $2.10 coffees) are kept.
- **ML categorization** with a confidence score on every prediction.
- **Review queue**: the least confident predictions come first. Correcting one can label every transaction from the same merchant, and retraining updates the rest.
- **Dashboard**: monthly income, spending, savings rate, a category breakdown and detected recurring charges (subscriptions, rent, bills).
- **Budgets**: monthly limits per category, spend vs. limit and a projection to month end.
- **Demo data**: one click generates 12 months of realistic synthetic transactions.

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/zanerwrobertson-cmd/zfinance.git
cd zfinance
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py             # the app
python -m pytest                 # the tests
python scripts/evaluate.py       # the benchmark; writes reports/evaluation.md
```

In the app, open **Import** and click **Load demo data** to try it without your own bank data. `sample_data/` also has example RBC and TD exports.

The database is created at `data/zfinance.db`, which git ignores.

## Architecture

```
app.py                    Streamlit UI: Import, Review, Dashboard, Budgets
zfinance/
  importers.py            CSV parsing: header detection, headerless inference, manual mapping
  normalize.py            Cleans raw bank descriptions into merchant strings
  db.py                   SQLite store: transactions, budgets, model run history
  categorizer.py          Merchant memory + TF-IDF / logistic regression model, evaluation
  analytics.py            Monthly summaries, category breakdown, budgets, recurring charges
  baseline.py             Keyword-rule baseline used for comparison
  synth.py                Synthetic transaction generator
scripts/evaluate.py       Benchmark against the baselines
tests/                    pytest suite
```

**Data flow:** CSV → `importers` (one schema: date, description, amount, account) → `normalize` → `db` (rows hashed so re-imports are skipped) → `categorizer` → `analytics` → UI.

### Importing

Each bank exports differently. Rather than hard-coding every format, the importer runs two passes:

1. **Header detection** finds a header row in the first few lines and maps known column names (date, description, amount, debit, credit) onto the schema.
2. **Headerless inference** works out each column from its content: which parses as dates, which is mostly text, which are numeric, and whether two numeric columns behave like a mutually exclusive debit/credit pair (TD and CIBC).

If both fail, the app shows a preview and lets you pick the columns.

### Normalization

`"POS PURCHASE SQ *GOLDEN BAKERY 0423"` and `"TIM HORTONS #4521 LONDON ON"` carry a lot of noise. The normalizer strips payment-processor prefixes, store numbers, dates, masked card numbers, terminal IDs and province suffixes. That way the same merchant always produces the same string.

### Categorizer

The categorizer has two layers:

1. **Merchant memory**: if you've already labelled this exact merchant, it reuses your most recent label.
2. **Model** for everything else: word and character n-gram TF-IDF over the cleaned description, plus amount features (sign, log magnitude, round-number flag), feeding a class-balanced multinomial logistic regression. The character n-grams let it generalize to merchants it has never seen, for example "golden bakery" → Dining.

User labels are never overwritten by predictions. Each retrain logs its cross-validated scores to the database.

### Evaluation methodology

A random train/test split lets the same merchant appear on both sides, so it mostly measures memorization. zfinance also reports **GroupKFold grouped by merchant**, where every test merchant is absent from training, and shows both numbers so the gap is visible. The benchmark also simulates real use: label the first N months, then predict the rest.

