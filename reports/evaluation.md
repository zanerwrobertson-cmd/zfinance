# zfinance evaluation

Averaged over 5 simulated users (~674 transactions, ~137 distinct merchant strings each).

## Overall accuracy (5-fold cross-validation)

| Method | Accuracy |
|---|---|
| Majority class | 35.1% |
| Keyword rules | 74.5% |
| Model, random split | 96.9% |
| Model, unseen merchants only | 54.0% (macro-F1 0.31) |

The random split lets the same merchant appear in train and test, which flatters the model. The unseen-merchant score (GroupKFold by normalized merchant) is the honest measure of generalization.

## Realistic use: label the first N months, predict the rest

Auto-accept threshold: confidence >= 0.8.

| Months labelled | Model accuracy | Keyword rules | Auto-categorized | Precision on auto-categorized |
|---|---|---|---|---|
| 1 | 77.7% | 74.7% | 44.0% | 100.0% |
| 2 | 86.2% | 74.3% | 60.1% | 99.8% |
| 3 | 91.5% | 74.4% | 70.8% | 99.9% |
| 6 | 96.6% | 74.4% | 88.5% | 99.9% |

Caveat: the data is synthetic and the keyword rules were written with knowledge of the merchant list, which favours the baseline. Re-run with `--db` on your own labelled data for real numbers.
