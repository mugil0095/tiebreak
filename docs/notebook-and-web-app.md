# Notebook helper and local web app

[Back to the README](../README.md)

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
