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

## Evaluation

Output of `python scripts/evaluate.py`, averaged over 5 simulated users (~674 transactions and ~137 distinct merchant strings each).

| Method | Accuracy |
|---|---|
| Majority class | 35.1% |
| Keyword rules | 74.5% |
| Model, random split | 96.9% |
| Model, unseen merchants only | 54.0% (macro-F1 0.31) |

**Realistic use: label the first N months, predict the rest** (auto-accept at confidence ≥ 0.8)

| Months labelled | Model accuracy | Keyword rules | Auto-categorized | Precision on auto-categorized |
|---|---|---|---|---|
| 1 | 77.7% | 74.7% | 44.0% | 100.0% |
| 2 | 86.2% | 74.3% | 60.1% | 99.8% |
| 3 | 91.5% | 74.4% | 70.8% | 99.9% |
| 6 | 96.6% | 74.4% | 88.5% | 99.9% |

What the numbers show:

- After one month of labels the model already edges out hand-written rules. After six months it reaches 96.6%.
- Predictions above the confidence threshold are 99.8% precise or better, so most transactions can be auto-accepted and only the uncertain ones need review.
- On merchants it has never seen, the model gets about 54%. The merchant memory and the review loop cover that gap.

**Caveat:** the data is synthetic, and the keyword rules were written knowing the merchant list, which favours the baseline. To benchmark your own labelled data, run `python scripts/evaluate.py --db data/zfinance.db`.

## Tests

```bash
python -m pytest
```

The tests cover description normalization, the RBC, TD and generic headerless importers, sign inversion, manual mapping, dedup on re-import, protection of user labels, and the categorizer (training guards, merchant memory, evaluation).
