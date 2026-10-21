# How tiebreak decides, and how it was validated

[Back to the README](../README.md)

## How the decision works

- **Equivalence margin.** You choose a margin (default 0.02 on the primary metric). A candidate is *tied* with the best only when we are 95% confident (Holm-adjusted for the number of comparisons) that it is no more than the margin below the best. "Not significantly different" is never treated as "tied".
- **Paired bootstrap.** The same resampled examples are scored for every candidate, 10,000 times with a fixed seed. Each resample recomputes every candidate's metric from its predictions, and which candidate is best, so the best is not assumed in advance. Holm's step-down correction handles the several comparisons against the best.
- **Discordance guard.** For classification metrics, a bootstrap alone has almost no variance when two models rarely disagree, so it could call near-identical models tied from very little evidence. The guard looks at the examples where the best is right and the candidate wrong, and the reverse, with a Bayesian prior (Dirichlet with +2 pseudo-counts). The tie test uses the more conservative of the two p-values. When models rarely disagree, tiebreak asks for more examples instead of declaring a tie.
- **Recommendation.** Among the candidates confirmed tied with the best, tiebreak picks the cheapest at your expected volume (training cost amortised plus per-prediction cost).
- **Provisional.** If the run is incomplete (cost cap) or a candidate was left out (provider errors, too many failures), the recommendation is marked PROVISIONAL and names what is missing. An example only counts as evaluated if at least one call for it succeeded, so provider errors are never scored as wrong answers; re-run with `--resume` to retry them.
- **"Cannot separate."** If a cheaper candidate looks within the margin but the data cannot confirm it, the report says so, names a safe choice, and gives an approximate number of extra labelled examples needed (this takes the guard into account and is labelled as an approximation).

## Statistical validation

`pytest -m slow` runs simulations against known truth (correlated candidates with exactly known accuracies). These cover accuracy and macro-F1 on two or more candidates only; regression metrics and ML candidates are not simulated, apart from the judge-score row. Measured, seeded rates:

| Check | Measured |
|---|---|
| 95% interval coverage (2 candidates, n=200) | 0.95 |
| False tie, true gap 0.20 | 0.000 |
| False tie at the exact margin boundary, n=100 / n=300 | 0.000 / 0.020 |
| False tie at the boundary, macro-F1 (3 classes), n=100 / n=300 | 0.000 / 0.010 |
| False tie at the boundary, non-nested errors (both models make unique errors), n=100 / n=300 | 0.000 / 0.000 |
| False tie at the boundary, judge scores (integers 1-5, gap = margin 0.25, bootstrap only), n=20 / n=100 | 0.061 / 0.067 |
| Early elimination drops a tied model, 5 tied candidates, n=120 / n=200 | 0.047 / 0.077 |
| Power: 2 truly equal models tied, n=800, margin 0.03 | 0.575 |

These are Monte-Carlo estimates from a few hundred replications each. They are a regression gate for `stats.py`, not guarantees.
