# zfinance evaluation

Averaged over 5 simulated users (~674 transactions, ~137 distinct merchant strings each).

## Overall accuracy (5-fold cross-validation)

| Method | Accuracy |
|---|---|
| Majority class | 35.1% |
| Keyword rules | 74.5% |
| Model, random split | 96.9% |
| Model, unseen merchants only | 54.0% (macro-F1 0.31) |

The random split lets the same merchant appear in train and test, which flatters the model. The unseen-merchant score (GroupKFold by merchant key) is the honest measure of generalization.

## Realistic use: label the first N months, predict the rest

Auto-accept threshold: confidence >= 0.8.

Memory = answered by merchant memory (exact or merchant-key match); model-only = everything else. Accuracy (all), Auto-categorized and Precision on auto-categorized combine both.

| Months labelled | Accuracy (all) | Keyword rules | Auto-categorized | Precision on auto-categorized | From memory | Memory accuracy | Model-only accuracy | Precision on auto-accepted model-only |
|---|---|---|---|---|---|---|---|---|
| 1 | 77.8% | 74.7% | 58.3% | 99.2% | 56.7% | 99.2% | 49.4% | 100.0% |
| 2 | 86.3% | 74.3% | 73.8% | 99.5% | 71.0% | 99.7% | 53.3% | 95.4% |
| 3 | 91.6% | 74.4% | 80.9% | 99.9% | 77.9% | 100.0% | 61.9% | 98.8% |
| 6 | 96.7% | 74.4% | 91.8% | 99.9% | 87.8% | 99.9% | 72.8% | 98.9% |

Caveat: the data is synthetic and the keyword rules were written with knowledge of the merchant list, which favours the baseline. Re-run with `--db` on your own labelled data for real numbers.
