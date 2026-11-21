# Safety switches and known limitations

[Back to the README](../README.md)

## Safety

- `--yes-send-data` is required before any hosted provider receives example text; without it, a run with hosted candidates refuses to start.
- `--local-only` refuses to run any hosted provider.
- `--max-cost USD` stops the run cleanly when the cap is reached, keeping completed work. The cap is cumulative across resumes (earlier spend counts even after you edit a prompt). It can overshoot by up to `concurrency - 1` in-flight calls. If a model's price is unknown the cap cannot be enforced; this is detected after the first paid call(s) and the run then stops with an error: add the model to `pricing.yaml`. Ctrl-C also stops spending (exit code 130, with a `--resume` hint).
- `--redact` (on `tiebreak report`) hides example ids in the report.
- API keys come from environment variables only; they are not written to run folders (the config snapshot redacts secrets). Recorded errors are scrubbed of the values of the `api_key_env` variables and of every set environment variable whose name ends in `_API_KEY`, `_TOKEN`, `_SECRET`, `_PASSWORD` or `APIKEY` (values of 8+ characters), plus `key=` query-string values. Any `base_url` or `api_base` that is not localhost marks a candidate as hosted, for every provider.
- Predictions are cached in `predictions.jsonl`. A cached answer is reused only if its prompt hash and model config hash still match, so editing a prompt or a model setting re-queries. `--resume <run folder>` continues a crashed or capped run without repeating paid calls.

## Limitations

- Tasks need ground-truth labels, or a judge model (see above): judge scores are only as good as the judge.
- Small samples are often inconclusive; the report then says "cannot separate" instead of guessing.
- With 4 or more similar candidates the tool needs MANY examples to confirm ties. Measured: 4 equal models need roughly 2000 examples at a 3-point margin, because best-of-k selection plus Holm correction is conservative.
- Early elimination of paid models only runs at 120 or more examples, and it is strict. Even so, with 5 truly tied candidates (margin 0.02, shipped settings) it dropped a tied model in 4.7% of simulated runs at n=120 and 7.7% at n=200 (300 seeded runs each).
- The discordance guard is accuracy-based, even when the metric is macro-F1.
- Mean absolute error and judge scores use a t-test guard (a paired t-test on the per-example score differences, with a variance prior); `neg_rmse` has no guard, so its tie test relies on the bootstrap alone and near-identical predictions are tied at any sample size. In one simulated setup (two correlated candidates scored 1 to 5, true gap equal to the margin) the guarded test falsely tied the worse candidate 1.7% of the time at n=20 and 4.7% at n=100; the bootstrap alone gave 6.1% and 6.7%. Other score distributions are not simulated. Scores that are constant for every candidate (a judge that gives every answer the same score) leave nothing to measure and stay tied; the report warns when that happens.
- For regression the margin is in the label's units (for `neg_rmse`/`neg_mae`), so the default 0.02 is rarely meaningful: set it yourself. A prediction that cannot be parsed is scored as maximally wrong.
- The few-shot pool (`shots_pool`) is class-balanced for classification tasks (every class appears, at least 2 examples per class stay for scoring) and random otherwise.
- `xgboost` needs every configured label to be present in the training data.
- Unknown or unpriced models are shown as the most expensive in the cost ranking.
- The `mock` provider supports classification tasks only.
- Latency is measured on your machine and network; it is not a provider benchmark.
- Prices change and are supplied by you; unknown prices are flagged in the report.
- Results hold for data that resembles your sample.
