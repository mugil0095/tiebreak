# Prompts without right answers, prices and history

[Back to the README](../README.md)

## Prompts without right answers (judge)

No labels? Give tiebreak prompts only. Every candidate answers every prompt, a **judge model** scores each answer, and the same tie test picks the cheapest candidate tied with the best.

```yaml
# task.yaml
type: judge
prompt_column: prompt
scale: [1, 5]            # the judge scores 1 (worst) to 5 (best)
margin: 0.25             # "tied" means within 0.25 score points of the best
# rubric: "Prefer short, correct, practical answers."    # optional
# human_score_column: human   # optional: your own scores for the FIRST candidate's answers
```

```yaml
# models.yaml: candidates as usual, plus exactly one judge
- {name: brief, kind: llm, provider: ollama, model: "<local model tag>", params: {think: false, max_tokens: 60}}
- {name: full, kind: llm, provider: ollama, model: "<local model tag>", params: {think: false, max_tokens: 300}}
- {name: judge, kind: llm, provider: typesafe, model: jev-1.14.0, role: judge}
```

```bash
tiebreak run --task examples/judge/task.yaml --models examples/judge/models.yaml --data examples/judge/prompts.csv   # offline demo, SIMULATED models
tiebreak run --task task.yaml --models models.yaml --data prompts.csv --yes-send-data                                # a hosted judge needs this flag
```

**Judges.** `provider: typesafe` uses TypeSafe's Jev through a `Score` question (`pip install "tiebreak[judge]"`, then set `TYPESAFE_API_KEY` in your environment yourself; tiebreak only checks that it is set and never writes it to the run folder). Jev is billed per input token only ($0.042 per million, output free, per TypeSafe's docs on 2026-10-18), so judging a few hundred answers costs cents. Any other provider (Ollama, LiteLLM, an OpenAI-compatible server) works as a judge too: it is asked for a single integer. There is no default judge: you choose one.

**What the report adds.** A Judge section states that the scores come from a model and not from ground truth, and shows the judge's cost (counted separately from each candidate's cost and included in `--max-cost`), the share of unusable replies and low-confidence judgments, each candidate's answer length and how strongly score follows length, a warning if the judge model is also a candidate, and, if you supplied `human_score_column`, the Spearman correlation between the judge and your own scores.

**Limits to know.**
- A judge has biases (longer answers, its own style or output). Score 30 to 50 answers yourself and check the agreement before trusting small differences.
- Jev's quality as a judge of open-ended answers has not been measured here; the human check is how you find out.
- Each candidate answers each prompt once. If the judge fails on any answer, that candidate is left out of the comparison (and the recommendation is marked provisional) until you `--resume`.
- The tie test for judge scores has no discordance guard (see Limitations).

## Prices and history

```bash
tiebreak models gpt-4o-mini        # input/output price per million tokens, from LiteLLM's bundled table
tiebreak history                   # every model you have run: score, cost, latency, from your runs/ folder
tiebreak history --csv > my-benchmark.csv
```

`tiebreak models` with no query lists the built-in estimators as before. Prices in a run come from your `pricing.yaml`, then the cost the provider reports; the lookup is for shortlisting candidates before you spend anything. `history` is your own benchmark: it only contains models you ran, on your tasks.
