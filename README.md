# RouteGuard — Cost-Aware Abstention for Text Classification

[Project website](https://sachinkg12.github.io/RouteGuard/) ·
[Artifact status](#artifact-and-publication-status) ·
[License](LICENSE)

[![tests](https://github.com/sachinkg12/RouteGuard/actions/workflows/tests.yml/badge.svg)](https://github.com/sachinkg12/RouteGuard/actions/workflows/tests.yml)

> **Publication status:** accepted at IEEE ICTAI 2026. The DOI and archival
> IEEE Xplore URL are not yet available and will be added after publication.
>
> **Frozen ICTAI artifact:** prepared for archival deposit; DOI pending.
> Paper-specific predictions and result records are intentionally not stored in Git.

A research and evaluation framework for **selective prediction (abstention) in text
classification**. It measures whether a classifier — classical or LLM-prompted —
should be allowed to **defer inputs that do not clear a decision threshold** instead
of always auto-deciding, and quantifies the trade-off under an explicit, asymmetric
cost model. A score can be well calibrated yet still rank correct and incorrect
predictions poorly, so the repository does not equate confidence with reliability.

The running example is **IT support ticket routing**: route a ticket to the right
team automatically when the model is confident, otherwise defer it to human triage.
A wrong auto-route is far more expensive than a deferral, so the interesting
question is not raw accuracy but **expected cost per item**.

---

## What it evaluates

For every model it reports accuracy / macro-F1 together with a selective-prediction evaluation view:

- **Calibration** — Expected Calibration Error (ECE), with optional isotonic recalibration.
- **Abstention curves** — coverage vs. routed-accuracy across confidence thresholds.
- **Cost** — expected cost per item under a configurable `(c_ok, c_triage, c_wrong)` model.
  The paper reports the illustrative `(0, 1, 5)` setting and threshold sensitivity;
  it does not estimate an organization's real deployment costs.
- **Error analysis** — confusion matrix and per-class breakdown for the best model.

ECE measures average calibration error; it does not establish that a confidence
signal ranks correct predictions above incorrect ones. Calibration claims should
therefore be read alongside confidence-discrimination or selective-risk analysis.

Everything is produced by one reproducible pipeline that snapshots its config, seeds,
dataset hash, and package versions alongside the results.

The scoped camera-ready headline, controlling manifest, verification receipt, and
text-free prediction vectors will be published together in the versioned artifact
DOI. This repository remains the reusable software record; see
[Artifact and publication status](#artifact-and-publication-status) for the
release boundary and verification workflow.

### Questions it answers

1. Can a classifier route reliably when allowed to defer uncertain cases to human review?
2. Does confidence-aware abstention reduce the wrong-decision rate at an acceptable coverage cost?
3. How sensitive is the route-or-defer conclusion to threshold resolution and
   alternative rejection scores?
4. How do classical baselines (TF-IDF + LR / SVM / RF, fine-tuned DistilBERT) compare to
   prompted LLMs in this asymmetric-cost regime?

---

## Architecture

The pipeline is unidirectional and separates prediction, confidence, abstention,
and cost evaluation:

```
Predictor → PredictionBatch → ConfidenceEstimator → AbstentionPolicy → CostModel → result bundle
```

Predictors never know about confidence methods; confidence methods never know about
abstention; and the evaluator never inspects predictor internals. Predictor
registration is additive: `tests/test_registry_open_closed.py` registers a fake
predictor at runtime and the evaluator handles it without modification. Other
extension types currently require explicit builder or policy-set wiring, as shown
below; the repository does not claim they are modification-free.

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
.venv/bin/pip install -e '.[test]'
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

- **[IT Service Ticket Classification Dataset, version 1](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset/versions/1)**
  (CC0, 47,837 rows, columns `Document` / `Topic_group`; source verified
  2026-09-05). Download it and place the CSV at
  `data/it_service_tickets.csv`. The loader is column-mapped, so adapting to a
  variant is a YAML change.

```bash
# sanity-check the loader + stratified split
.venv/bin/python scripts/prepare_data.py --config configs/experiment_default.yaml

# headline classical run (TF-IDF + LR, full test set, no API calls)
.venv/bin/python scripts/run_classification.py --config configs/experiment_default.yaml

# select the operating threshold on calibration, then evaluate it once on test
.venv/bin/python scripts/select_threshold_on_calibration.py

# additional classical baselines (Linear SVM, Random Forest) and fine-tuned DistilBERT
.venv/bin/python scripts/run_classification.py --config configs/experiment_classical_extras.yaml
.venv/bin/python scripts/run_classification.py --config configs/experiment_distilbert.yaml
```

The synthetic quickstart validates installation and pipeline behavior; it does
**not** reproduce the paper's empirical results. Exact reproduction uses the
separately archived DOI artifact containing the frozen manifest, receipt, aggregate
tables, and text-free vectors. Re-running hosted LLM endpoints today is a new
experiment because provider behavior can change. See
[Artifact and publication status](#artifact-and-publication-status) for the
release boundary, and the section below for the download layout and verification
commands.

### Reproduce the frozen ICTAI record

**Artifact DOI: pending.** After publication, download the version-specific DOI
archive and extract it into the repository root so that `artifacts/ictai2026/`
exists locally. That directory is gitignored. Then run:

```bash
.venv/bin/python scripts/verify_camera_ready_manifest.py \
  --paper-dir artifacts/ictai2026
.venv/bin/python scripts/verify_public_claims.py \
  --artifact-dir artifacts/ictai2026
.venv/bin/python scripts/baselines_on_llm_subsample.py \
  --input artifacts/ictai2026/table_ii_common_subset.json
```

The first command also requires dataset version 1 at
`data/it_service_tickets.csv`. The public-claim check and Table II reconstruction
make no hosted-model API calls.

`run_classification.py` reports the configured threshold sweep. Use
`select_threshold_on_calibration.py` for the deployment-style protocol: choose
the threshold on calibration data, then report its test result without selecting
on test.

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

Hosted-model names do not freeze provider behavior. Record the provider, resolved
model identifier, request date, prompt/config hash, and API cost for every archival
LLM run.

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

`outputs/` and `artifacts/` are intentionally gitignored. GitHub carries the living
implementation; the versioned DOI archive carries the frozen camera-ready evidence.
Do not describe the synthetic run as reproduction of the paper's numbers; see
[Artifact and publication status](#artifact-and-publication-status).

---

## Tests

```bash
.venv/bin/pytest                       # whole suite
.venv/bin/pytest -k abstention -v      # by keyword
```

Covers stratified-split correctness, train/calibration/test disjointness,
few-shot-from-train-only, threshold abstention, cost-model math, ECE on toy
distributions, LLM-parser strictness on malformed JSON and out-of-set labels,
the registry-based predictor swap (proof for predictor registration only), and an
end-to-end bundle smoke test.

---

## Project structure

```
src/ticket_routing/
  data/        # loaders, stratified splitter, schema
  models/      # Predictor interface + registry + TF-IDF (LR/SVM/RF) + DistilBERT + LLM-prompted classifier
  llm/         # LLMClient interface + mock / openai-compatible / anthropic / local-transformers + prompt + parser
  confidence/  # ModelReported / SelfConsistency / Agreement + optional Platt & isotonic recalibration
  abstention/  # AlwaysRoute / Threshold + experimental Agreement policy code path
  evaluation/  # metrics, ECE, cost model, evaluator, error analysis
  reporting/   # tables, plots, notes, bundle writer
  utils/       # config, hashing, logging, seeds, versions
configs/       # experiment YAMLs (synthetic debug, classical, DistilBERT, LLM)
scripts/       # prepare_data, run_classification, run_cost_analysis, analysis utilities
tests/         # split, metrics, abstention, cost, parser, ECE, registry, smoke
outputs/       # run dirs (gitignored)
```

The paper's reported agreement analysis uses cross-predictor agreement as a
confidence signal. The experimental `AgreementPolicy` code path should not be read
as a reported non-trivial abstention-policy comparator.

---

## Constraints

- **Public datasets only.** No private or proprietary data.
- **Reproducibility.** Every run snapshots its config, seeds, dataset hash, and package versions.
- **Scope.** An evaluation framework, not a production service — no serving layer, dashboards, or DB.

## Reported camera-ready findings

These numbers are a compact orientation, not a substitute for the paper's full
methods, tables, uncertainty analysis, and interpretation:

- On the 47,837-ticket, eight-class corpus, the four trained non-LLM baselines
  reach test accuracy from 0.818 to 0.877. The best of eight prompted settings
  across four LLM families reaches 0.477 on the common 1,000-ticket subset.
- Under the illustrative `(0, 1, 5)` cost model, TF-IDF + Logistic Regression at
  the calibration-selected threshold `0.60` reduces test cost from `0.763` to
  `0.472`: 38.1% lower (paired-bootstrap 95% CI 35.7–40.4%), with 70.5% coverage,
  95.0% routed accuracy, and a 3.5% wrong-route rate.
- At the analytically derived LLM break-even threshold `0.80`, seven of eight
  prompted settings route 0 of 1,000 tickets. Haiku few-shot routes 19 of 1,000,
  all correctly, for cost `0.981` versus `1.000` for defer-all. This is a
  low-coverage observation, not evidence of broad auto-routing readiness.
- Raw prompted-LLM ECE spans 0.332–0.594 and isotonic ECE spans 0.014–0.049, but
  lower ECE does not guarantee useful ranking. GPT-4o-mini few-shot has correctness
  AUROC 0.500 both before and after isotonic calibration.

The website is an explanatory companion, the repository is the reusable software,
and the artifact DOI will be the frozen evidence package. After publication, IEEE
Xplore will be the authoritative paper record and citation destination.

## Artifact and publication status

This repository contains reusable code, configurations, tests, and documentation.
The paper-specific numerical evidence is intentionally distributed as a separate,
immutable archival artifact. It has been frozen and audited locally; its DOI is
pending. No DOI, Git tag, arXiv identifier, or IEEE Xplore URL is claimed before it
exists.

### Public now

| Item | Status | Verification path |
|---|---|---|
| Source code | Available | Repository source and commit history |
| Project website | Available | <https://sachinkg12.github.io/RouteGuard/> |
| Experiment configurations | Available | `configs/` |
| Tests and synthetic smoke run | Available | `tests/` and `configs/experiment_small_debug.yaml` |
| Calibration-selected threshold check | Available | `scripts/select_threshold_on_calibration.py` |
| Evaluation dataset | Available upstream, not redistributed | [IT Service Ticket Classification Dataset, version 1](https://www.kaggle.com/datasets/adisongoh/it-service-ticket-classification-dataset/versions/1), CC0 |
| Per-run hashes and environment metadata | Generated locally | Each `paper_bundle/` produced by the pipeline |

### Frozen DOI artifact

**Artifact DOI: pending.** The prepared archive contains:

- the scoped public claim record;
- the complete audited manifest at `camera_ready_results/claim_manifest.json` and
  its 86-check verification receipt;
- aggregate camera-ready result tables;
- sanitized, text-free LLM prediction/confidence vectors;
- the sanitized Table II common-subset inputs;
- the aggregate near-duplicate diagnostic; and
- checksums and reproducibility instructions.

The archive is held outside this Git repository until archival publication. See
[Reproduce the frozen ICTAI record](#reproduce-the-frozen-ictai-record) for the
download layout and verification commands.

### Remaining publication actions

| Item | Current status | Completion condition |
|---|---|---|
| Downloadable supplemental archive | Prepared and audited locally; public release pending | Deposit the final public-safe archive in the DOI-bearing archival record |
| Versioned software release | Pending | Tag the exact camera-ready commit after final verification |
| Archival artifact DOI | Pending | Deposit the frozen release and result bundle in an archival repository |
| Accepted-manuscript/arXiv record | Prepared locally; submission and identifier pending | Submit only the author-accepted/preprint version permitted by IEEE policy, with the required notice |
| IEEE Xplore URL and paper DOI | Pending publication | Add only after IEEE assigns the archival record |

### Claim provenance

The website's interactive chart uses frozen TF-IDF + Logistic Regression test
operating points at thresholds 0.50 through 0.90. It is a descriptive sensitivity
illustration: it must not be used to select a deployable threshold on the test set.
The paper protocol selects the operating point using calibration data and evaluates
it once on the disjoint test set.

The DOI artifact's public claim record contains a deliberately small, scoped subset
of the camera-ready manifest and names that manifest's SHA-256. The artifact also
contains the complete public-safe manifest and its verification receipt. The paper
remains the scholarly account; the website and README are navigation and
reproduction aids, not substitutes for the IEEE record. The synthetic smoke run
proves that the pipeline executes; it does not reproduce the paper's empirical
claims.

### Publication boundary

- No ticket text, raw prompt/response logs, private review-time archive, author
  proof, or IEEE Version of Record is included.
- GitHub contains no paper-specific prediction vectors or frozen result package.
  Those records are included only in the DOI artifact to support verification and
  reproduction.
- After publication, add the paper DOI and canonical IEEE Xplore URL to
  `README.md`, `CITATION.cff`, and the website. Do not substitute an author proof
  or an unauthorized copy of the Version of Record.
- The outer supplemental ZIP hash is intentionally not embedded in files packaged
  inside that ZIP; its checksum belongs in the release/deposit metadata.

## Citation

This is the companion code for:

> Sachin Gupta. *Cost-Aware Abstention for LLM-Based IT Ticket Classification.*
> IEEE International Conference on Tools with Artificial Intelligence (ICTAI), 2026.
> Accepted; archival record pending.

```bibtex
@inproceedings{gupta2026costaware,
  author    = {Sachin Gupta},
  title     = {Cost-Aware Abstention for {LLM}-Based {IT} Ticket Classification},
  booktitle = {Proceedings of the IEEE International Conference on Tools with
               Artificial Intelligence (ICTAI)},
  year      = {2026},
  note      = {Accepted; archival IEEE Xplore record pending}
}
```

Machine-readable software and preferred paper citation metadata are available in
[`CITATION.cff`](CITATION.cff). Add the DOI and IEEE Xplore URL there and above only
after they have been assigned.

## License

- **Code:** MIT (see `LICENSE`).
- **Dataset:** upstream CC0 dedication; the data is not redistributed here.
- **Manuscript and paper-derived material:** governed by the applicable IEEE
  electronic copyright form. No IEEE Version of Record is hosted in this repo.

This repository and its GitHub Pages site are independent project materials, not
official IEEE or ICTAI websites. Author-posting guidance is maintained by the
[IEEE Author Center](https://conferences.ieeeauthorcenter.ieee.org/author-ethics/guidelines-and-policies/submission-policies/).
