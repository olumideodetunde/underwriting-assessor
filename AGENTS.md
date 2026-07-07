# Underwriting Assessor — Agent Guide

Project-intrinsic instructions for any agent working in this repo.
Read this before trusting `README.md` or `PROJECT_STRUCTURE.md` (see the drift warning below).

## What this is

An ML-powered insurance underwriting assessor that predicts premium pricing from historical data and risk factors.
It decomposes pricing the actuarial way: Pure Premium = Frequency × Severity.
Frequency (how often claims occur) is modelled with Poisson regression; severity (how costly a claim is) with Gamma regression.

The design separates two pricing scenarios: new clients (blind risk, no claims history) and renewals (informed risk, with history).
Behaviour is config-driven: the same training code runs both scenarios, and the only difference is which YAML config is passed (`config/new_client.yaml` vs `config/renewal.yaml`).

## Heads-up: docs vs reality

`README.md` and `PROJECT_STRUCTURE.md` describe an idealised, partly-planned layout that does NOT match the current code.
Trust the code and this file over those two documents.
Known mismatches:

| Docs claim | Actual code |
|---|---|
| `src/serving/app.py`; endpoints `/predict/new_client`, `/health` | `src/app/main.py`; endpoints `POST /predict`, `GET /ready` |
| `scripts/train.py`; `make train-new`, `make serve`, `make test`, `make lint` | `scripts/` holds only `gate_metrics_smoke.py` (no `train.py`); run `python -m src.train.frequency`; Makefile has only terraform/docker targets |
| `src/features/` (plural), `src/monitoring.py`, `config/base.yaml` | `src/feature/` (singular); no `monitoring.py`; only `new_client.yaml` + `renewal.yaml` |
| `src/training/trainer.py` | `src/train/frequency.py` and `src/train/severity.py` |
| Dual serving endpoints, one per model | The app currently serves a SINGLE frequency model on `/predict` (title: "Claims Frequency Inference") |

Do not invent the `make` targets or paths the README implies; verify against the code.

## Build / test / run

```bash
# Install (UV package manager, Python >=3.8.1)
uv venv && source .venv/bin/activate
uv pip install -e ".[test]"

# Test (config lives in pyproject.toml: pythonpath=["."], testpaths=["tests"])
pytest tests/ -v
pytest --cov=src

# Train the frequency model (uses config/renewal.yaml via its __main__ block)
python -m src.train.frequency
# Severity model logic lives in src/train/severity.py

# Serve locally (loads an MLflow model from S3 at startup)
cd src/app && uvicorn main:app --host 0.0.0.0 --port 8000
# Requires env vars MODEL_BUCKET_NAME and MODEL_PATH (see .env)

# Docker (image exposes port 80; healthcheck hits GET /ready)
cd src/app && docker build -t underwriting-assessor .

# Deploy to AWS (manual; terraform + ECR/ECS)
make setup-ecr          # bootstrap ECR via terraform/setup
make deploy-container    # build + push image (src/app/deploy.sh)
make deploy-service      # ECS/ALB via terraform/app
```

There is no `make test`, `make serve`, `make train`, or `make lint` target — those exist only in the docs.

## Review gate (no-mistakes)

`no-mistakes` is the mandatory review gate for this repo.
Every agent routes finished work through it - agents do NOT `git push origin` or open PRs by hand.
The gate runs an AI-driven pipeline (review → test → docs → lint) in a disposable worktree, auto-applies safe fixes, escalates judgment calls, and opens a clean PR against `origin` only when every check is green.
The user's only manual step is reviewing that final gated PR.

- **Standard issue-to-PR workflow**: the `/task-to-pr` skill (`.claude/skills/task-to-pr/SKILL.md`) captures the full loop - pick an issue, branch in an isolated worktree off a clean `main`, implement and test, commit with repo conventions, then hand the branch to the gate.
Invoke it when starting work on a GitHub issue or when unsure of the branch/worktree/PR flow.
- **Gate your work** when a task's changes are committed on a branch: run `/no-mistakes` (agent skill - gates existing committed work and drives the pipeline headlessly) or `git push no-mistakes <branch>`.
Do not `git push origin` and do not open a PR yourself; the gate does both.
- **What the gate runs here** (from `.no-mistakes.yaml`): `pytest tests/ -v` then `python -m scripts.gate_metrics_smoke` as the test step.
The lint step is a deterministic no-op (`true`) because no linter is wired (see the note in `## Conventions & gotchas`); do not invent one.
- **Data-science evidence on the PR**: `scripts/gate_metrics_smoke.py` retrains the frequency model on the committed dataset (`data/input/Motor_vehicle_insurance_data.csv`, via `config/smoke.yaml`) with no MLflow/S3, printing the metrics table (MSE/RMSE/MAE/R²/Poisson deviance) and saving diagnostic plots under `.no-mistakes/evidence/`.
It reuses `train_and_evaluate()` in `src/train/frequency.py` (the tracking-free half of `run()`).
- **Setup prerequisite**: the gate must be initialised once per clone with `no-mistakes init` (creates the `no-mistakes` remote and installs the `/no-mistakes` skill).
If the `no-mistakes` remote or `/no-mistakes` skill is missing, run `no-mistakes init` before gating.
- Honour the existing conventions: plain `-` not `—`, and never auto-add an agent as a commit co-author - the gate authors the PR, so agents must not hand-push or hand-write PR bodies.

## Architecture / pipeline flow

```
config/renewal.yaml (or new_client.yaml)
        │
        ▼  src/config.py         validate config (Pydantic)
        ▼  src/data/loader.py    load CSV (';' delimiter, na_values ["NA",""])
        ▼  src/data/splitter.py  train/test split (default random_state 42)
        ▼  src/feature/          FittedFeaturePipeline.fit(train).transform(...) (Vehicle then Driver)
        ▼  src/model/factory.py  select_training_algorithm(name, params)
        ▼  src/train/frequency.py orchestrate: fit → predict → metrics → plots
        ▼  src/metrics.py        MSE/RMSE/MAE/R²/Poisson deviance + plots
        ▼  src/tracking.py       log params/metrics/model/figures to MLflow
        ▼  S3                    model artifact (MLflow format)
        ▼  src/app/main.py       FastAPI: POST /predict, GET /ready
```

Stubs / not yet implemented: `src/feature/claims.py`, `src/feature/payment.py`, `src/storage.py`.
The model factory maps algorithm strings to estimators: `xgboost`, `poisson_regressor`, `gamma_regressor`, `gradient_boosting`.

## Conventions & gotchas

- Feature transformers follow a fit-on-train / transform contract to prevent data leakage (`src/feature/base.py`).
`Vehicle.fit(trainset)` learns its fuel-type encoding on the training split only, then `.transform()` applies it to train and test.
`Driver` gained a `fit()` for interface consistency even though its features are row-wise; keep both transformers fit/transform-shaped.
`FittedFeaturePipeline` (`src/feature/pipeline.py`, exported from `src/feature/__init__.py`) composes the two into one fit/transform unit - `fit()` learns state via `Vehicle.fit`, `transform()` applies Vehicle then Driver in training order and raises `RuntimeError` if called before `fit()`.
Frequency training now uses it: `train_and_evaluate()` in `src/train/frequency.py` fits the pipeline on the train split and transforms both splits through it (issue #112).
MLflow persistence of the fitted pipeline is still deferred to issue #115.
- Config drives behaviour: the YAML is the only difference between the new-client and renewal models — put new knobs in config, not in branching code.
- Notebooks explore, `src/` implements: exploratory work lives in `notebook/`; production code with tests lives in `src/`.
- Type hints are expected throughout (modern style: `dict[str, Any]`, `list[str]`, `X | None`).
- No linter/formatter is wired up yet: the docs mention `ruff`, but there is no ruff/black/pre-commit config in the repo.
- Raw CSV is semicolon-delimited with `"NA"`/`""` treated as missing (`src/data/loader.py`).
- MLflow runs against a local tracking server by default (`http://127.0.0.1:5000`); runtime config comes from `.env` (`MODEL_BUCKET_NAME`, `MODEL_PATH`, AWS settings).
- CI (`.github/workflows/docker-build-deploy.yml`) only builds and pushes the Docker image on push to `main`; it does not run tests or lint.

## Working style

The user's global `~/.claude/CLAUDE.md` conventions apply here.
In particular: never use the em dash "—" (use a plain "-"); prefer quality, robustness, and long-term maintainability over minimising development cost; reproduce a bug end-to-end before fixing it; and never auto-add an agent name as a commit co-author.
