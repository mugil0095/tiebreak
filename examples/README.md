# Real examples

Three real comparisons on public data. They cost nothing to run: two scikit-learn models plus a random forest, and one small local LLM (Qwen3.5 0.8B through Ollama). The reports are on the [sample reports page](https://mugil0095.github.io/tiebreak/).

```bash
python examples/fetch_data.py        # downloads AG News, SMS Spam and SST-2, writes 300-row samples (seed 0) to examples/data/
ollama pull qwen3.5:0.8b
tiebreak run --task examples/agnews/task.yaml --models examples/models.yaml --data examples/data/agnews.csv --local-only
```

Task files: `agnews/`, `spam/`, `sentiment/`; candidates: `models.yaml`. All use macro-F1 with a 0.05 margin and a volume of 10,000 predictions.

## Results (macro-F1, 95% CI; n = 300 labelled rows each)

| Task | logreg | boosted | forest | Qwen3.5 0.8B (local) | Recommended |
|---|---|---|---|---|---|
| AG News (4 topics) | **0.692** [0.638, 0.742] | 0.374 [0.318, 0.427] | 0.563 [0.506, 0.616] | 0.358 (eliminated early) | logreg |
| SMS spam (37 of 300 are spam) | 0.467 [0.456, 0.477] | **0.778** [0.694, 0.850] | 0.688 [0.589, 0.777] | 0.130 (eliminated early) | boosted |
| SST-2 sentiment (balanced) | 0.597 [0.541, 0.652] | 0.534 [0.477, 0.590] | 0.593 [0.537, 0.647] | **0.794** [0.746, 0.838] | Qwen3.5 0.8B |

## What these runs show

- **The winner changes with the task.** Classic ML won two tasks and the small LLM won the third. No single model is "best", which is the point of measuring.
- **Nothing was tied.** In all three runs the best candidate was clearly ahead, so the recommendation is simply the best one. For a run where a cheaper model ties a more expensive one, see the logistic regression vs Gemini result in the main README.
- **Early elimination works as designed.** On AG News and spam the LLM was far behind after 100 of 300 examples, so tiebreak stopped calling it. Its score covers those 100 examples only.
- **The small LLM labels every SMS "spam".** On the first 100 messages it processed (15 spam, 85 ham) it answered "spam" every time, so 85 answers were wrong. That is real model behaviour with this prompt, not a parsing problem.

## Caveats

- 300 rows is small, and every interval above is wide. One sample per task.
- scikit-learn models use near-default settings on a TF-IDF text representation, trained on only a few hundred rows. The unweighted logistic regression predicts almost everything as "ham" on the imbalanced spam sample (0.467). A tuned baseline would do better.
- Only one small local LLM was tested. These runs say nothing about larger models.
- The local LLM's cost is latency times a compute price per hour, so it is an estimate, not a bill.
- Runs used a GTX 1650, so LLM latency (about 0.8 s median per call) is specific to that machine.
