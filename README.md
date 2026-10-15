# tiebreak

**Which model should you use for your task?** tiebreak runs several candidate models (hosted LLM APIs, local LLMs via Ollama, and classic scikit-learn models) on a small labelled sample of *your* data and reports score, cost and latency with 95% confidence intervals. It then recommends the **cheapest model that is statistically tied with the best**, or says plainly that your data cannot separate the candidates and roughly how many more labelled examples would. Everything runs on your machine and produces a self-contained offline `report.html`.

## Two-minute demo

```bash
pip install -e ".[dev]"
tiebreak init --demo --dir demo
tiebreak run --task demo/task.yaml --models demo/models.yaml --data demo/data.csv
```

(`python -m tiebreak ...` works the same way.) The command prints the path of `report.html`; open it in a browser.

**The demo LLMs are SIMULATED.** The `sim-*` candidates use the `mock` provider: fake models with a preset accuracy that make no API calls. The report carries a SIMULATED banner. The demo only shows how tiebreak works; never use its results to choose a real model.

What the demo currently produces: `sim-large` has the highest score, but **`sim-small` is recommended**, because it is confirmed to be within the 0.05 margin of `sim-large` and is much cheaper.

## Use your own data

Put your labelled examples in a CSV or JSONL file (one row per example; an id column is optional, a label column is required), then describe the task and the candidates.

```yaml
# task.yaml
type: classify            # classify | binary | extract_exact | regress
label_column: sentiment
text_column: review
labels: [negative, neutral, positive]
metric: macro_f1
margin: 0.02
volume: 10000
```

```yaml
# models.yaml
- name: hosted-a
  kind: llm
  provider: anthropic            # any LiteLLM provider id (needs: pip install "tiebreak[llm]")
  model: "<model id from your account>"
- name: local-small
  kind: llm
  provider: ollama
  model: "<local model tag>"
- name: logreg
  kind: ml
  estimator: logistic_regression
- name: boosted
  kind: ml
  estimator: hist_gradient_boosting
```

Model ids are configuration strings; none are hard-coded. API keys are read from environment variables only. Prices come from an optional `pricing.yaml` that you supply (`--pricing`).

```bash
tiebreak estimate --task task.yaml --models models.yaml --data mydata.csv   # dry run, calls nothing
tiebreak run --task task.yaml --models models.yaml --data mydata.csv --yes-send-data
tiebreak report runs/<run>        # regenerate the report, no model calls
tiebreak diagnose runs/<run>      # suspected labelling mistakes, disagreements
```

## How the decision works

- **Equivalence margin.** You choose a margin (default 0.02 on the primary metric). A candidate is *tied* with the best only when we are 95% confident (Holm-adjusted for the number of comparisons) that it is no more than the margin below the best. "Not significantly different" is never treated as "tied".
- **Paired bootstrap.** The same resampled examples are scored for every candidate, 10,000 times with a fixed seed. Each resample recomputes every candidate's metric from its predictions, and which candidate is best, so the best is not assumed in advance. Holm's step-down correction handles the several comparisons against the best.
- **Discordance guard.** For classification metrics, a bootstrap alone has almost no variance when two models rarely disagree, so it could call near-identical models tied from very little evidence. The guard looks at the examples where the best is right and the candidate wrong, and the reverse, with a Bayesian prior (Dirichlet with +2 pseudo-counts). The tie test uses the more conservative of the two p-values. When models rarely disagree, tiebreak asks for more examples instead of declaring a tie.
- **Recommendation.** Among the candidates confirmed tied with the best, tiebreak picks the cheapest at your expected volume (training cost amortised plus per-prediction cost).
- **"Cannot separate."** If a cheaper candidate looks within the margin but the data cannot confirm it, the report says so, names a safe choice, and gives an approximate number of extra labelled examples needed (this takes the guard into account and is labelled as an approximation).

## Safety

- `--yes-send-data` is required before any hosted provider receives example text; without it, a run with hosted candidates refuses to start.
- `--local-only` refuses to run any hosted provider.
- `--max-cost USD` stops the run cleanly when the cap is reached, keeping completed work. The cap is cumulative across resumes. If a model's price is unknown the cap cannot be enforced, so the run refuses to start rather than risk overspending: add the model to `pricing.yaml`. Ctrl-C also stops spending.
- `--redact` (on `tiebreak report`) hides example ids in the report.
- API keys come from environment variables only; they are not written to run folders (the config snapshot redacts secrets) and are scrubbed from recorded errors.
- Predictions are cached in `predictions.jsonl`. A cached answer is reused only if its prompt hash and model config hash still match, so editing a prompt or a model setting re-queries. `--resume <run folder>` continues a crashed or capped run without repeating paid calls.

## Limitations

- Only tasks with ground-truth labels are supported (no open-ended generation judged by an LLM).
- Small samples are often inconclusive; the report then says "cannot separate" instead of guessing.
- With 4 or more similar candidates the tool needs MANY examples to confirm ties. Measured: 4 equal models need roughly 2000 examples at a 3-point margin, because best-of-k selection plus Holm correction is conservative.
- Early elimination of paid models only runs at 120 or more examples, and it is strict. Even so, with 5 tied candidates it still dropped a tied model in about 8% of simulated runs.
- The discordance guard is accuracy-based, even when the metric is macro-F1.
- Regression metrics have no discordance guard, so near-identical regressors can be declared tied regardless of sample size.
- Latency is measured on your machine and network; it is not a provider benchmark.
- Prices change and are supplied by you; unknown prices are flagged in the report.
- Results hold for data that resembles your sample.

## Statistical validation

`pytest -m slow` runs simulations against known truth (correlated candidates with exactly known accuracies). Measured, seeded rates:

| Check | Measured |
|---|---|
| 95% interval coverage (2 candidates, n=200) | 0.95 |
| False tie, true gap 0.20 | 0.000 |
| False tie at the exact margin boundary, n=100 / n=300 | 0.000 / 0.020 |
| Power: 2 truly equal models tied, n=800, margin 0.03 | 0.575 |

These are Monte-Carlo estimates from a few hundred replications each. They are a regression gate for `stats.py`, not guarantees.

## Contributing

```bash
pip install -e ".[dev]"
pytest                    # fast suite, no network
pytest -m slow            # statistical simulations
ruff check .
```

Live provider tests only run with `TIEBREAK_LIVE=1`. Licensed under Apache-2.0.
