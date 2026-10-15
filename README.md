# llm-eval-harness

An eval and tracing harness for LLM features, in Python (standard library only; pytest for tests).
It runs a dataset through a model, scores each output with **deterministic checks plus an LLM-as-judge**,
traces every step, **checks the judge against your own labels**, and **fails the build when a prompt or
model change makes things worse**.

It ships with two tasks:

| Task | Data | Purpose |
|---|---|---|
| `pubmedqa` | **Real public data:** PubMedQA PQA-L, 1,000 expert-annotated yes/no/maybe questions answered from PubMed abstracts (MIT licence, Jin et al. 2019) | Evaluate a health-research QA step: correctness, JSON format, grounded rationale, no personal advice |
| `openers` | 10 authored cases | The offline demo (mock models) that shows the regression gate working without an API key |

> **Honest status.** The PubMedQA loader and task are fully unit-tested on a small **hand-written format fixture**
> (`tests/fixtures/*_FIXTURE.json`, clearly not real data), because the real file was not downloadable in the
> environment where this was built. **No results on the real dataset or a live model are claimed here.** The first
> step below produces them. The loader accepts the original release layout and the Hugging Face export layout, and
> stops with a clear message if the file looks different.

## What is in the harness

| Piece | Where | What it gives you |
|---|---|---|
| Versioned datasets + schema | `evalkit/schema.py` | Strict validation (missing/unexpected fields, types, enums, duplicate ids). A sidecar `.meta.json` stores version + sha256, so editing a dataset without re-freezing **fails loudly**. |
| Real-data pipeline | `evalkit/pubmedqa_data.py`, `scripts/prepare_pubmedqa.py` | Normalise PubMedQA, drop unusable rows (and report how many), seeded sample (natural or balanced label mix), freeze with a version. |
| Deterministic checks | `evalkit/checks.py`, `evalkit/qa_checks.py` | Cheap, repeatable checks run before any judge. |
| LLM-as-judge | `evalkit/judge.py`, `evalkit/tasks.py` | Rubric with a strict JSON contract. An unparseable judge reply is **invalid** and the case fails visibly. |
| Judge calibration | `evalkit/calibration.py` | Export a sample to score by hand, then report exact agreement, within-1, bias, and Cohen's kappa. This is how you defend the judge. |
| Step-level traces | `evalkit/tracing.py` | Every step logs latency, tokens, estimated cost, input/output, errors (JSONL). |
| Async runner | `evalkit/runner.py` | Concurrent cases, semaphore limit, per-call timeout, provider errors captured per case. |
| Real API client | `evalkit/providers.py` | Retries 429/5xx with exponential backoff; temperature 0 by default (`EVALKIT_TEMPERATURE`, empty string omits it). |
| Regression gate | `evalkit/regression.py` | Newly failing cases, pass-rate drop, judge-mean drop, with **explicit tolerances** for noisy real models. Refuses to compare across different datasets. Non-zero exit for CI. |

## Run it on the real dataset

```
# 1. Get the data (MIT licence). Either the official release file from https://github.com/pubmedqa/pubmedqa
#    (ori_pqal.json) or an export of the "pqa_labeled" subset from Hugging Face. Put it in data_raw/.

# 2. Build a frozen, versioned sample (100 cases, seed 7, natural label mix)
python scripts/prepare_pubmedqa.py --input data_raw/ori_pqal.json --n 100 --seed 7 --version 1.0.0

# 3. Baseline with prompt v1 (set your key; never commit it)
export ANTHROPIC_API_KEY=...
python -m evalkit run --task pubmedqa --dataset datasets/pubmedqa_sample.jsonl \
  --prompt prompts/pubmedqa_v1.txt --provider anthropic --judge anthropic \
  --save-run runs/v1.json --save-baseline baselines/pubmedqa_v1.json

# 4. Change the prompt (v2) and let the gate decide. Tolerances absorb normal run-to-run noise.
python -m evalkit run --task pubmedqa --dataset datasets/pubmedqa_sample.jsonl \
  --prompt prompts/pubmedqa_v2.txt --provider anthropic --judge anthropic \
  --baseline baselines/pubmedqa_v1.json --max-new-failures 3 --pass-rate-tolerance 0.03 --save-run runs/v2.json

# 5. Calibrate the judge: export 30 outputs, score them yourself (1-5), compare
python -m evalkit label-sheet --run runs/v1.json --out labels.csv --n 30
#    ... fill the human_score column ...
python -m evalkit agreement --labels labels.csv

# 6. Read the traces: traces/<run_id>.jsonl, filter on trace_id "<run_id>:<pubmed id>"
# 7. Fill in docs/RESULTS_TEMPLATE.md with YOUR numbers and commit it.
```
Each run makes about two API calls per case (answer + judge). Start with `--n 30` to check cost.
Set `EVALKIT_MODEL` to change the model. Cost figures are estimates from configurable per-million-token rates.

## Offline demo (no key, no data)
```
python -m pytest -q
python -m evalkit run --dataset datasets/openers_v1.jsonl --schema datasets/schema_openers.json \
  --prompt prompts/opener_v1.txt --provider mock:regressed --baseline baselines/openers_v1_good.json
# -> pass rate 100% -> 40%, names the 6 failing cases, exit code 1
```

On Windows PowerShell, run the same commands from the `llm-eval-harness` directory:
```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m evalkit run --dataset datasets/openers_v1.jsonl --schema datasets/schema_openers.json `
  --prompt prompts/opener_v1.txt --provider mock:regressed --baseline baselines/openers_v1_good.json
```
For live Anthropic runs, set the key for the current PowerShell session with
`$env:ANTHROPIC_API_KEY = "..."`; do not save it in the project or commit it.

## A typical week, as this supports it
An output looks wrong -> open its trace -> see which step failed -> add the case to the dataset -> bump the
version and re-freeze -> the gate guards against it from then on.

## Verification
- 76 tests: schema, checks, QA checks, tracing, judge parsing, async runner, concurrency, timeouts, retry/backoff,
  PubMedQA loader and sampling, regression tolerances, calibration maths, CLI exit codes.
- I deliberately broke the code in 7 places (retries, grounding check, the two regression rules, kappa maths, enum
  validation, sample ordering); the tests caught each one. Two regression rules initially slipped through, which
  exposed untested paths, so I added tests for them.
- pytest could not be installed in the build environment, so tests were run with a small stand-in runner. They are plain
  pytest-style functions; run `python -m pytest -q` to confirm on your machine.
- Not verified: the real Anthropic call path against the live API, and the loader against the real PubMedQA file.

## Known limits
- Single model family, small samples (100 cases gives wide error bars, especially for the rare "maybe" label).
- Judge not calibrated until you run step 5; accuracy on a research benchmark is not clinical safety.
- Grounding check is a word-overlap heuristic; it catches off-topic rationales, not subtle unsupported claims (that is the judge's job).
- Retries are simple exponential backoff without jitter.

## Layout
```
evalkit/    schema, checks, qa_checks, tasks, providers, judge, tracing, runner, regression, calibration, pubmedqa_data, CLI
scripts/    prepare_pubmedqa.py
datasets/   schemas, openers demo data (frozen), pubmedqa_sample.* appears after step 2
prompts/    opener_v1/v2, pubmedqa_v1 (grounded, allows "maybe") and v2 (told to be decisive: a deliberate test of the gate)
baselines/  saved baselines
docs/       RESULTS_TEMPLATE.md
tests/      76 tests + a hand-written format fixture (not real data)
```
