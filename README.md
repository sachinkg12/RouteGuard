# RouteGuard — Cost-Aware Abstention for Text Classification

A research and evaluation framework for **selective prediction (abstention) in text
classification**. It measures whether a classifier — classical or LLM-prompted —
should be allowed to **defer low-confidence inputs to human review** instead of
always auto-deciding, and quantifies the trade-off under an explicit, asymmetric
cost model.

The running example is **IT support ticket routing**: route a ticket to the right
team automatically when the model is confident, otherwise defer it to human triage.
A wrong auto-route is far more expensive than a deferral, so the interesting
question is not raw accuracy but **expected cost per item**.

---

## What it evaluates

For every model it reports not just accuracy / macro-F1 but the full selective-prediction picture:

- **Calibration** — Expected Calibration Error (ECE), with optional isotonic recalibration.
- **Abstention curves** — coverage vs. routed-accuracy across confidence thresholds.
- **Cost** — expected cost per item under a configurable `(c_ok, c_triage, c_wrong)` model,
  swept over a grid of cost settings for robustness.
- **Error analysis** — confusion matrix and per-class breakdown for the best model.

Everything is produced by one reproducible pipeline that snapshots its config, seeds,
dataset hash, and package versions alongside the results.

### Questions it answers

1. Can a classifier route reliably when allowed to defer uncertain cases to human review?
2. Does confidence-aware abstention reduce the wrong-decision rate at an acceptable coverage cost?
3. Is the relative cost reduction from deferral robust across plausible cost settings?
4. How do classical baselines (TF-IDF + LR / SVM / RF, fine-tuned DistilBERT) compare to
   prompted LLMs in this asymmetric-cost regime?

---

## Architecture

The pipeline is unidirectional and the evaluator is closed for modification:

```
Predictor → PredictionBatch → ConfidenceEstimator → AbstentionPolicy → CostModel → result bundle
```

Predictors never know about confidence methods; confidence methods never know about
abstention; the evaluator never inspects predictor internals. Adding a model,
confidence method, abstention policy, dataset, or LLM client is **additive** — no
existing file changes. `tests/test_registry_open_closed.py` is the executable proof:
it registers a fake predictor at runtime and the evaluator handles it with no code change.

| Extension | Where | How |
|---|---|---|
| New model | `src/ticket_routing/models/your_model.py` | Implement `Predictor`, decorate with `@register("name")` from `models.registry` |
| New confidence method | `src/ticket_routing/confidence/your_method.py` | Subclass `ConfidenceEstimator`, wire into `_build_confidence` in `scripts/run_classification.py` |
| New abstention policy | `src/ticket_routing/abstention/your_policy.py` | Subclass `AbstentionPolicy`, add to `Evaluator._build_policies` |
| New dataset | `src/ticket_routing/data/loaders.py` | Implement `DatasetLoader.load() -> DatasetBundle`, add a branch in `build_loader_from_config` |
| New LLM client | `src/ticket_routing/llm/your_client.py` | Implement `LLMClient.generate(prompt) -> LLMResponse`, add a branch in `build_client_from_config` |

---

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e .[test]
```

Optional extras: `.[llm-local]` pulls in `transformers` + `torch` for the DistilBERT
baseline and local HuggingFace models.

---

## Quickstart

```bash
# 60-second end-to-end run on the bundled synthetic generator — no downloads, no API keys
.venv/bin/python scripts/run_classification.py --config configs/experiment_small_debug.yaml

# inspect the output bundle
ls outputs/debug_synth__*/paper_bundle/
```

The bundle contains the result tables (CSV + markdown), figures (PNG), and run metadata.

> The synthetic loader is for tests and smoke runs only — never treat its numbers as real results.

---

## Running on the real dataset

The evaluation dataset is public and **not** included in this repo (`data/` is gitignored).

- **IT Service Ticket Classification Dataset** (public, 47,837 rows, columns
  `Document` / `Topic_group`). Download it and place the CSV at
  `data/it_service_tickets.csv`. The loader is column-mapped, so adapting to a
  variant is a YAML change.

```bash
# sanity-check the loader + stratified split
.venv/bin/python scripts/prepare_data.py --config configs/experiment_default.yaml

# headline classical run (TF-IDF + LR, full test set, no API calls)
.venv/bin/python scripts/run_classification.py --config configs/experiment_default.yaml

# additional classical baselines (Linear SVM, Random Forest) and fine-tuned DistilBERT
.venv/bin/python scripts/run_classification.py --config configs/experiment_classical_extras.yaml
.venv/bin/python scripts/run_classification.py --config configs/experiment_distilbert.yaml
```

### LLM runs

```bash
# copy .env.example to .env and fill in provider keys first
.venv/bin/python scripts/run_classification.py --config configs/experiment_llm_sample.yaml
```

- The default client is a no-op **mock** (`DryRunMockClient`); no external calls happen
  unless a config explicitly opts in.
- LLM predictors run on a stratified subsample of size `dataset.max_test_samples_for_llm`
  to bound API cost; the classical model still evaluates on the full test set.
- To use a local model, set `client.kind: local_transformers` and `client.model` to a
  HuggingFace id (requires the `llm-local` extra).

See `.env.example` for the supported providers (OpenRouter single-key, or direct
OpenAI / Anthropic / Together / Groq keys).

---

## Output bundle

Each run writes `outputs/<run_id>__<timestamp>/paper_bundle/`:

| Artifact | Path |
|---|---|
| Dataset summary | `tables/table_1_dataset_summary.csv` |
| Classification performance | `tables/table_2_classification.csv` |
| Calibration (ECE) | `tables/table_3_calibration.csv` |
| Abstention (coverage / routed accuracy) | `tables/table_4_abstention.csv` |
| Cost-aware routing | `tables/table_5_cost.csv` |
| Error analysis | `tables/table_6_error_analysis.csv` |
| Figures | `figures/*.png` (class distribution, coverage-vs-accuracy, cost-vs-threshold, confusion, pipeline) |
| Run summary | `result_summary.md` |
| Reproducibility statement (dataset hash, seed, split sizes, package versions) | `reproducibility_statement.md` |

Rebuild the cost sweep from a saved run without re-predicting:

```bash
.venv/bin/python scripts/run_cost_analysis.py --run-dir outputs/<run>
```

---

## Tests

```bash
.venv/bin/pytest                       # whole suite
.venv/bin/pytest -k abstention -v      # by keyword
```

Covers stratified-split correctness, train/calibration/test disjointness,
few-shot-from-train-only, threshold abstention, cost-model math, ECE on toy
distributions, LLM-parser strictness on malformed JSON and out-of-set labels,
the registry-based predictor swap (open/closed proof), and an end-to-end bundle smoke test.

---

## Project structure

```
src/ticket_routing/
  data/        # loaders, stratified splitter, schema
  models/      # Predictor interface + registry + TF-IDF (LR/SVM/RF) + DistilBERT + LLM-prompted classifier
  llm/         # LLMClient interface + mock / openai-compatible / anthropic / local-transformers + prompt + parser
  confidence/  # ModelReported / SelfConsistency / Agreement + optional Platt & isotonic recalibration
  abstention/  # AlwaysRoute / Threshold / Agreement policies
  evaluation/  # metrics, ECE, cost model, evaluator, error analysis
  reporting/   # tables, plots, notes, bundle writer
  utils/       # config, hashing, logging, seeds, versions
configs/       # experiment YAMLs (synthetic debug, classical, DistilBERT, LLM)
scripts/       # prepare_data, run_classification, run_cost_analysis, analysis utilities
tests/         # split, metrics, abstention, cost, parser, ECE, registry, smoke
outputs/       # run dirs (gitignored)
```

---

## Constraints

- **Public datasets only.** No private or proprietary data.
- **Reproducibility.** Every run snapshots its config, seeds, dataset hash, and package versions.
- **Scope.** An evaluation framework, not a production service — no serving layer, dashboards, or DB.

## Citation

This is the companion code for:

> Sachin Gupta. *Cost-Aware Abstention for LLM-Based IT Ticket Classification.*
> IEEE International Conference on Tools with Artificial Intelligence (ICTAI), 2026. To appear.

```bibtex
@inproceedings{gupta2026costaware,
  author    = {Sachin Gupta},
  title     = {Cost-Aware Abstention for {LLM}-Based {IT} Ticket Classification},
  booktitle = {Proceedings of the IEEE International Conference on Tools with
               Artificial Intelligence (ICTAI)},
  year      = {2026},
  note      = {To appear}
}
```

## License

Code under MIT (see `LICENSE`). The evaluation dataset is governed by its own
upstream terms and is not redistributed here.
