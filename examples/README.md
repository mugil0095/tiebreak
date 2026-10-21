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

## Judge example (prompts only)

`examples/judge/` compares candidates on 20 prompts with no right answers. `models.yaml` runs offline with simulated models (it is also a test). `models-ollama.yaml` runs for free with one small local model playing all three roles, and `models-jev.yaml` swaps in Jev as the judge.

A real run of `models-ollama.yaml` (Qwen3.5 0.8B as two candidates, "brief" with a 60-token limit and "full" with 300, and also as the judge, on a machine with a GTX 1650):

| | Mean judge score (1-5) | Mean answer length | Median latency per answer |
|---|---|---|---|
| qwen-brief | 5.00 | 42 words | 5.0 s |
| qwen-full | 5.00 | 167 words | 18.5 s |

- **The 0.8B judge gave all 40 answers a 5.** It cannot tell the candidates apart, so tiebreak reported them as tied and recommended the cheaper one (qwen-brief), and said plainly in the report that the judge scored every answer the same and that this tie says nothing about quality. It also warned that the judge is one of the candidates.
- **That is the point of the warnings.** A weak or self-interested judge produces confident-looking ties. Use a stronger judge (for example Jev) and score some answers yourself (`human_score_column`) before trusting a judge-based recommendation.
- Not measured here: how Jev behaves as a judge on these prompts.

### The same prompts with Jev as the judge

The same candidates judged by Jev (`models-jev.yaml`, `jev-1.14.0`), on the same 20 prompts:

| | Mean judge score (1-5) | 95% interval | Mean answer length | Median latency per answer |
|---|---|---|---|---|
| qwen-brief | 3.08 | 2.83 to 3.40 | 42 words | 4.6 s |
| qwen-full | **3.55** | 3.15 to 3.94 | 167 words | 13.9 s |

- **Jev tells the answers apart.** Scores ran from 2.06 to 4.92 (standard deviation 0.65 for brief, 0.90 for full), where the 0.8B local judge gave every answer a 5. tiebreak recommended qwen-full: its lead of 0.46 points is bigger than the 0.25 margin, but the intervals overlap, so treat it as a lean, not a result.
- **It costs almost nothing.** 40 judgments used 22,925 input tokens: $0.00096 in total, which matches 22,925 x $0.042 per million.
- **Longer was not better here.** For qwen-full the score fell as answers got longer (Spearman -0.73). In the lowest- and highest-scored answers I read (6 of 40), Jev rewarded short, correct answers to prompts that asked for brevity ("under 60 words", "two sentences") and marked down long or cut-off answers to them. That is a reading of a handful of answers, not a validation.
- **Not validated against people.** Only 1 of 40 judgments had confidence below 0.5, but nobody has compared these scores with human scores yet. Add a `human_score_column` for the first candidate's answers to see the agreement in the report.
