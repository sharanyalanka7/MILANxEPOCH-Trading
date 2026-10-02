# Multi-Asset Machine Learning Trading System

## Submission contents
- `Final_Trading_ML_Report.docx` — submission-ready report
- `fold_metrics.csv` — IC and RankIC by walk-forward fold
- `model_summary.csv` — headline model/portfolio metrics
- `ic_by_fold.png` — IC stability plot
- `rankic_by_fold.png` — RankIC stability plot
- `long_short_returns.png` — long-short cumulative-return comparison

## Experiment
- 14 assets
- 15,770 cleaned observations
- 17 engineered features
- 5-period forward-return target
- ExtraTrees, Random Forest, Gradient Boosting
- Ensemble predictions
- 5-fold purged date-based walk-forward validation

## Important metric note
This is a regression/ranking trading experiment, so conventional classification accuracy is not the main metric. IC, RankIC, portfolio return, Sharpe ratio, win rate and drawdown are used instead.

## Validation
- 5 folds
- No duplicate prediction rows
- No overlapping evaluation dates
