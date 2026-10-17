<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" alt="tiebreak" width="360">
  </picture>
</p>

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

Model ids are configuration strings; none are hard-coded. API keys are read from environment variables only. Prices: your optional `pricing.yaml` (`--pricing`) wins; otherwise the cost the provider reports per call is used; otherwise the price is unknown. Each candidate's `price_source` (`pricing.yaml`, `provider`, `unknown` or `compute time`) is shown in the report with a "Prices used" section and the run date. Local models (Ollama, or an endpoint on localhost) are costed from call latency times `compute_price_per_hour`, like scikit-learn models.

```bash
tiebreak estimate --task task.yaml --models models.yaml --data mydata.csv   # dry run, calls nothing
tiebreak run --task task.yaml --models models.yaml --data mydata.csv --yes-send-data
tiebreak report runs/<run>        # regenerate the report, no model calls
tiebreak diagnose runs/<run>      # suspected labelling mistakes, disagreements
```

### Reasoning ("thinking") models

tiebreak asks for short answers (`temperature: 0`, `max_tokens: 64`). A reasoning model can spend all 64 tokens on hidden thinking and return an empty answer; tiebreak then reports a high invalid rate and says so. Override the defaults per candidate through `params`:

```yaml
- {name: qwen-small, kind: llm, provider: ollama, model: "qwen3.5:0.8b", params: {think: false}}
- {name: hosted-r, kind: llm, provider: <provider>, model: "<model id>", params: {max_tokens: 2048}}
```

`params` are passed straight to LiteLLM and replace the defaults; they are part of the cache key, so changing them re-runs the calls.

### Rate limits (free tiers)

Hosted free tiers allow only a few requests per minute. Set `requests_per_minute` on a candidate and tiebreak spaces its calls, and use `--concurrency 1` so calls do not burst:

```yaml
- {name: hosted-a, kind: llm, provider: <provider>, model: "<model id>", requests_per_minute: 10}
```

If a provider still answers 429, tiebreak waits 8, 16, 32 then 60 seconds between up to six retries (other errors retry with a 1, 2, 4 second backoff). A 429 that names a per-day quota is not retried (waiting minutes cannot clear it): the call is recorded with that reason and the warning shows it, so use a model with a bigger daily quota or come back after the reset. Calls that still fail are recorded as errors, the candidate is excluded with a warning, and `--resume` retries only those calls.

### Verify with your own provider

The demo models are simulated, so confirm a real provider end to end with a tiny, capped run (needs `pip install "tiebreak[llm]"` and your own key):

```yaml
# my-models.yaml: one real hosted candidate plus a free baseline
- name: hosted-a
  kind: llm
  provider: <provider>              # a LiteLLM provider id, e.g. the one your key belongs to
  model: "<model id from your account>"
  api_key_env: MY_PROVIDER_API_KEY  # name of the environment variable that holds your key
- name: logreg
  kind: ml
  estimator: logistic_regression
```

```bash
export MY_PROVIDER_API_KEY=...      # PowerShell: $env:MY_PROVIDER_API_KEY = "..."
tiebreak estimate --task task.yaml --models my-models.yaml --data mydata.csv    # calls nothing; check the cost
tiebreak run --task task.yaml --models my-models.yaml --data mydata.csv --yes-send-data --max-cost 0.05
```

Use a small labelled sample (40 to 60 rows) and add the model to `pricing.yaml` (`--pricing`) if `estimate` reports its price as unknown, because the cap cannot be enforced without a price. The result will usually be inconclusive at that size; what you are checking is that the calls succeed, costs are recorded and the report opens.

## Notebook

```python
import tiebreak
c = tiebreak.compare(task, models, df)   # task and models: YAML paths, dicts or config objects; df: a DataFrame or a file path
c                                         # in Jupyter this shows the report
c.table                                   # the results as a DataFrame
c.save_report("r.html")                   # the same self-contained offline report
```

## Web app

```bash
pip install "tiebreak[web]"
tiebreak serve                 # http://127.0.0.1:8765 ; options: --port, --runs-dir
```

A single page where you paste the task and models, pick a data file (or click **Load demo**), press **Run**, watch progress, and read the report. What it does and does not do:

- It listens on this computer only (`127.0.0.1`) and refuses requests addressed to any other host or coming from another web site.
- API keys are read from environment variables, as in the CLI; the page never asks for or shows a key.
- One run at a time; a second one is refused until the first finishes. Past runs are kept in `--runs-dir` (default `runs/`) and listed on the page.
- Hosted candidates send example text to third parties, so the page has a consent box that must be ticked first (and a local-only box that refuses them).
- Only paste a models.yaml you wrote or trust: a candidate's base_url and api_key_env decide which server receives the key from your environment.
- There is no authentication: anyone logged in on the same computer can reach 127.0.0.1:PORT while it runs.
- It is not a hosted service and has no accounts or authentication; do not expose it to a network.

## How the decision works

- **Equivalence margin.** You choose a margin (default 0.02 on the primary metric). A candidate is *tied* with the best only when we are 95% confident (Holm-adjusted for the number of comparisons) that it is no more than the margin below the best. "Not significantly different" is never treated as "tied".
- **Paired bootstrap.** The same resampled examples are scored for every candidate, 10,000 times with a fixed seed. Each resample recomputes every candidate's metric from its predictions, and which candidate is best, so the best is not assumed in advance. Holm's step-down correction handles the several comparisons against the best.
- **Discordance guard.** For classification metrics, a bootstrap alone has almost no variance when two models rarely disagree, so it could call near-identical models tied from very little evidence. The guard looks at the examples where the best is right and the candidate wrong, and the reverse, with a Bayesian prior (Dirichlet with +2 pseudo-counts). The tie test uses the more conservative of the two p-values. When models rarely disagree, tiebreak asks for more examples instead of declaring a tie.
- **Recommendation.** Among the candidates confirmed tied with the best, tiebreak picks the cheapest at your expected volume (training cost amortised plus per-prediction cost).
- **Provisional.** If the run is incomplete (cost cap) or a candidate was left out (provider errors, too many failures), the recommendation is marked PROVISIONAL and names what is missing. An example only counts as evaluated if at least one call for it succeeded, so provider errors are never scored as wrong answers; re-run with `--resume` to retry them.
- **"Cannot separate."** If a cheaper candidate looks within the margin but the data cannot confirm it, the report says so, names a safe choice, and gives an approximate number of extra labelled examples needed (this takes the guard into account and is labelled as an approximation).

## Safety

- `--yes-send-data` is required before any hosted provider receives example text; without it, a run with hosted candidates refuses to start.
- `--local-only` refuses to run any hosted provider.
- `--max-cost USD` stops the run cleanly when the cap is reached, keeping completed work. The cap is cumulative across resumes (earlier spend counts even after you edit a prompt). It can overshoot by up to `concurrency - 1` in-flight calls. If a model's price is unknown the cap cannot be enforced; this is detected after the first paid call(s) and the run then stops with an error: add the model to `pricing.yaml`. Ctrl-C also stops spending (exit code 130, with a `--resume` hint).
- `--redact` (on `tiebreak report`) hides example ids in the report.
- API keys come from environment variables only; they are not written to run folders (the config snapshot redacts secrets). Recorded errors are scrubbed of the values of the `api_key_env` variables and of every set environment variable whose name ends in `_API_KEY`, `_TOKEN`, `_SECRET`, `_PASSWORD` or `APIKEY` (values of 8+ characters), plus `key=` query-string values. Any `base_url` or `api_base` that is not localhost marks a candidate as hosted, for every provider.
- Predictions are cached in `predictions.jsonl`. A cached answer is reused only if its prompt hash and model config hash still match, so editing a prompt or a model setting re-queries. `--resume <run folder>` continues a crashed or capped run without repeating paid calls.

## Limitations

- Only tasks with ground-truth labels are supported (no open-ended generation judged by an LLM).
- Small samples are often inconclusive; the report then says "cannot separate" instead of guessing.
- With 4 or more similar candidates the tool needs MANY examples to confirm ties. Measured: 4 equal models need roughly 2000 examples at a 3-point margin, because best-of-k selection plus Holm correction is conservative.
- Early elimination of paid models only runs at 120 or more examples, and it is strict. Even so, with 5 truly tied candidates (margin 0.02, shipped settings) it dropped a tied model in 4.7% of simulated runs at n=120 and 7.7% at n=200 (300 seeded runs each).
- The discordance guard is accuracy-based, even when the metric is macro-F1.
- Regression metrics have no discordance guard, so near-identical regressors can be declared tied regardless of sample size.
- For regression the margin is in the label's units (for `neg_rmse`/`neg_mae`), so the default 0.02 is rarely meaningful: set it yourself. A prediction that cannot be parsed is scored as maximally wrong.
- The few-shot pool (`shots_pool`) is a random sample, not stratified by class.
- `xgboost` needs every configured label to be present in the training data.
- Unknown or unpriced models are shown as the most expensive in the cost ranking.
- The `mock` provider supports classification tasks only.
- Latency is measured on your machine and network; it is not a provider benchmark.
- Prices change and are supplied by you; unknown prices are flagged in the report.
- Results hold for data that resembles your sample.

## Statistical validation

`pytest -m slow` runs simulations against known truth (correlated candidates with exactly known accuracies). These cover accuracy and macro-F1 on two or more candidates only; regression metrics and ML candidates are not simulated. Measured, seeded rates:

| Check | Measured |
|---|---|
| 95% interval coverage (2 candidates, n=200) | 0.95 |
| False tie, true gap 0.20 | 0.000 |
| False tie at the exact margin boundary, n=100 / n=300 | 0.000 / 0.020 |
| False tie at the boundary, macro-F1 (3 classes), n=100 / n=300 | 0.000 / 0.010 |
| False tie at the boundary, non-nested errors (both models make unique errors), n=100 / n=300 | 0.000 / 0.000 |
| Early elimination drops a tied model, 5 tied candidates, n=120 / n=200 | 0.047 / 0.077 |
| Power: 2 truly equal models tied, n=800, margin 0.03 | 0.575 |

These are Monte-Carlo estimates from a few hundred replications each. They are a regression gate for `stats.py`, not guarantees.

## Contributing

```bash
pip install -e ".[dev]"
pytest                    # fast suite, no network
pytest -m slow            # statistical simulations
ruff check .
```

Live provider tests are planned but do not exist yet; the suite never touches the network. Licensed under Apache-2.0.
