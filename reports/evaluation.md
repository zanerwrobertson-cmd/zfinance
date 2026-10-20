# zfinance evaluation

Averaged over 5 simulated users (~658 transactions, ~132 distinct merchant strings each).

## Overall accuracy (5-fold cross-validation)

| Method | Accuracy |
|---|---|
| Majority class | 20.9% |
| Keyword rules | 77.2% |
| Model, random split | 97.0% |
| Model, unseen merchants only | 50.2% (macro-F1 0.32) |

The random split lets the same merchant appear in train and test, which flatters the model. The unseen-merchant score (GroupKFold by normalized merchant) is the honest measure of generalization.

## Realistic use: label the first N months, predict the rest

Auto-accept threshold: confidence >= 0.8.

| Months labelled | Model accuracy | Keyword rules | Auto-categorized | Precision on auto-categorized |
|---|---|---|---|---|
| 1 | 77.2% | 77.1% | 47.7% | 99.7% |
| 2 | 85.0% | 76.8% | 62.4% | 99.8% |
| 3 | 89.4% | 77.0% | 71.7% | 99.6% |
| 6 | 94.7% | 78.2% | 87.1% | 99.8% |

Caveat: the data is synthetic and the keyword rules were written with knowledge of the merchant list, which favours the baseline. Re-run with `--db` on your own labelled data for real numbers.
