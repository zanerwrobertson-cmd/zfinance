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

## Deploying

To host a public demo on Streamlit Cloud, add this to the app's secrets (or set it as an environment variable):

```toml
ZFINANCE_PUBLIC = 1
```

In public mode each browser session gets its own temporary SQLite file, which is deleted when the session ends, and the app shows a banner warning visitors not to upload real bank exports. Without the setting, the app uses the shared `data/zfinance.db`, so never deploy it publicly that way.

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

1. **Merchant memory**: if you've already labelled this exact merchant string, it reuses your most recent label. Otherwise it falls back to the merchant key (the first two words, so "second cup st thomas" matches "second cup london"), but only when all your labels under that key agree. A key with conflicting labels, such as "e transfer" used for both Rent and Other, is left to the model. The Review page's "same merchant" option follows the same rule.
2. **Model** for everything else: word and character n-gram TF-IDF over the cleaned description, plus amount features (sign, log magnitude, round-number flag), feeding a class-balanced multinomial logistic regression. The character n-grams let it generalize to merchants it has never seen, for example "golden bakery" → Dining.

User labels are never overwritten by predictions. Each retrain logs its cross-validated scores to the database.

### Evaluation methodology

A random train/test split lets the same merchant appear on both sides, so it mostly measures memorization. zfinance also reports **GroupKFold grouped by merchant key**, where no test merchant shares even its first two words with training, and shows both numbers so the gap is visible. The benchmark also simulates real use: label the first N months, then predict the rest.

## Evaluation

Output of `python scripts/evaluate.py`, averaged over 5 simulated users (~674 transactions and ~137 distinct merchant strings each). The synthetic data uses fixed seeds (0-4) and a fixed end date (2026-09-30), so these numbers reproduce exactly.

| Method | Accuracy |
|---|---|
| Majority class | 35.1% |
| Keyword rules | 74.5% |
| Model, random split | 96.9% |
| Model, unseen merchants only | 54.0% (macro-F1 0.31) |

**Realistic use: label the first N months, predict the rest** (auto-accept at confidence ≥ 0.8)

Memory = answered by merchant memory (exact or merchant-key match); model-only = everything else. Accuracy (all), Auto-categorized and Precision on auto-categorized combine both.

| Months labelled | Accuracy (all) | Keyword rules | Auto-categorized | Precision on auto-categorized | From memory | Memory accuracy | Model-only accuracy | Precision on auto-accepted model-only |
|---|---|---|---|---|---|---|---|---|
| 1 | 77.8% | 74.7% | 58.3% | 99.2% | 56.7% | 99.2% | 49.4% | 100.0% |
| 2 | 86.3% | 74.3% | 73.8% | 99.5% | 71.0% | 99.7% | 53.3% | 95.4% |
| 3 | 91.6% | 74.4% | 80.9% | 99.9% | 77.9% | 100.0% | 61.9% | 98.8% |
| 6 | 96.7% | 74.4% | 91.8% | 99.9% | 87.8% | 99.9% | 72.8% | 98.9% |

What the numbers show:

- After one month of labels, overall accuracy (77.8%) is roughly even with the keyword rules (74.7%). After six months it reaches 96.7%, against 74.4% for the rules.
- Most of that comes from merchant memory. It answers 57% of transactions after one month and 88% after six, and those answers are over 99% accurate.
- The model on its own, on the transactions memory can't answer, gets 49.4% after one month and 72.8% after six. That's below the keyword rules, though the rules are scored on all transactions, not on this harder subset. On unseen merchants in cross-validation the model gets 54.0%.
- Auto-accepted predictions (confidence ≥ 0.8) are 99.2% precise or better, and they cover 58% of transactions after one month and 92% after six. Nearly all of them are memory hits: the model alone clears the threshold for only 1.6-4.0% of transactions, so its 95-100% precision there rests on a handful of transactions per user.

**Caveat:** the data is synthetic, and the keyword rules were written knowing the merchant list, which favours the baseline. To benchmark your own labelled data, run `python scripts/evaluate.py --db data/zfinance.db`.

## Tests

```bash
python -m pytest
```

The tests cover description normalization, the RBC, TD and generic headerless importers, sign inversion, manual mapping, dedup on re-import, protection of user labels, and the categorizer (training guards, exact and merchant-key memory, review-page label propagation, prediction sources, evaluation), and per-session database isolation.
