# Using tiebreak with your own data

[Back to the README](../README.md)

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
  provider: anthropic            # any LiteLLM provider id (needs: pip install "tiebreak[llm] @ git+https://github.com/mugil0095/tiebreak.git")
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

<details>
<summary><b>Reasoning ("thinking") models</b></summary>

tiebreak asks for short answers (`temperature: 0`, `max_tokens: 64`). A reasoning model can spend all 64 tokens on hidden thinking and return an empty answer; tiebreak then reports a high invalid rate and says so. Override the defaults per candidate through `params`:

```yaml
- {name: qwen-small, kind: llm, provider: ollama, model: "qwen3.5:0.8b", params: {think: false}}
- {name: hosted-r, kind: llm, provider: <provider>, model: "<model id>", params: {max_tokens: 2048}}
```

`params` are passed straight to LiteLLM and replace the defaults; they are part of the cache key, so changing them re-runs the calls.

</details>

<details>
<summary><b>Rate limits (free tiers)</b></summary>

Hosted free tiers allow only a few requests per minute. Set `requests_per_minute` on a candidate and tiebreak spaces its calls, and use `--concurrency 1` so calls do not burst:

```yaml
- {name: hosted-a, kind: llm, provider: <provider>, model: "<model id>", requests_per_minute: 10}
```

If a provider still answers 429, tiebreak waits 8, 16, 32 then 60 seconds between up to six retries (other errors retry with a 1, 2, 4 second backoff). A 429 that names a per-day quota is not retried (waiting minutes cannot clear it): the call is recorded with that reason and the warning shows it, so use a model with a bigger daily quota or come back after the reset. Calls that still fail are recorded as errors, the candidate is excluded with a warning, and `--resume` retries only those calls.

</details>

<details>
<summary><b>Verify with your own provider</b></summary>

The demo models are simulated, so confirm a real provider end to end with a tiny, capped run (needs `pip install "tiebreak[llm] @ git+https://github.com/mugil0095/tiebreak.git"` and your own key):

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

</details>
