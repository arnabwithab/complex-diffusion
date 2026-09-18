# AGENTS.md

## Project Overview

A/B experiment: complex-valued PCT-style attention vs real-valued sigmoid-attention control (~51M each) in a small masked diffusion LM. `docs/spec.md` is the source of truth — it wins over everything here. Task tracker: `docs/features.json`.

## Development Philosophy
- Spec first: changing experiment parameters requires explicit user approval.
- TDD first: write the test, then the implementation. Never skip.
- No function ships without a test (smoke-level at minimum: shapes, finiteness, param counts).
- Explicit over clever — readable code beats smart code.

## Tech Stack
- Python (3.12+), PyTorch, HuggingFace `transformers` / `datasets`.
- This is an experiment repo, not an app. No frontend, no server, no database, no deployment.

## Conventions

### Python
- **Package manager: `uv`** — never invoke `pip` directly.
- Formatter: `black`, Linter: `ruff` (includes import sorting).
- snake_case for everything — files, variables, functions.
- No `print` in training code paths — stdlib `logging` only.
- Hyperparameters mirror `docs/spec.md`. If code differs from the spec, the code is wrong.
- Both variants train under identical hypers. Never tune per-variant without explicit approval.
- Secrets (Kaggle keys, tokens): never in code, logs, or commits.

### General
- Commits: conventional commits format (feat:, fix:, chore:, docs:, test:, refactor:).
- If something feels out of scope, flag it rather than silently doing it.
