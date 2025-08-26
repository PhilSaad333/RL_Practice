# Repository Guidelines

## Planning With This Agent
- Purpose: I help you design high‑level plans, de‑risk changes, and define acceptance criteria before coding. Use `CLAUDE.md` for prior patterns; use this doc for the planning flow.
- Kickoff: share a short brief (e.g., "Add PPO for CartPole with TensorBoard, 2 seeds, 100k steps"). I return a sequenced plan, risks, and exit criteria.
- Plan cadence: I keep a living checklist (milestones, owners, ETA). Ask me to refine, split, or re‑order as context evolves. I summarize progress after merges or experiment runs.
- Change design: I draft minimal diffs and interfaces first (trainer hooks, config keys, logging fields), previewing file and directory changes to keep PRs small and traceable.
- Experiments: I propose configs, seeds, metrics, and success thresholds; I compare to baselines and summarize results with next actions.

## Project Structure & Module Organization
- Source: `src/` or `rl_practice/` (algorithms, env adapters, utils).
- Experiments: `experiments/` with scripts and `configs/*.yaml`.
- Tests: `tests/` mirrors package layout (e.g., `tests/algorithms/`).
- Artifacts: `runs/` or `outputs/` for logs, checkpoints, and metrics.
- Example: add `src/algorithms/dqn/` with a matching `tests/algorithms/test_dqn.py`.

## Build, Test, and Development Commands
- Setup (Windows): `python -m venv .venv && .venv\Scripts\activate && pip install -e .[dev]`
- Lint/format: `ruff check .` and `ruff format .` (or `black .`, `isort .` if used).
- Tests: `pytest -q` (subset: `pytest -q tests/algorithms`); coverage: `pytest -q --cov=rl_practice`.
- Run example: `python experiments/train.py --config experiments/configs/dqn_cartpole.yaml`.

## Coding Style & Naming Conventions
- Style: PEP 8, 4‑space indent, type hints for public APIs.
- Names: modules/functions/vars `snake_case`, classes `PascalCase`, constants `UPPER_CASE`.
- Imports: stdlib → third‑party → local; no wildcard imports. Concise Google/NumPy‑style docstrings.

## Testing Guidelines
- Framework: `pytest` (optionally `pytest-cov`). Target ≥85% coverage for core training/algorithms.
- Conventions: one test module per code module; name as `test_<module>.py` and `test_<behavior>`; seed RNGs for determinism.

## Commit & Pull Request Guidelines
- Commits: imperative with scoped prefix (e.g., `trainer: add early-stop hook`).
- PRs: include purpose, rationale, testing steps, linked issues, and sample logs/plots for training changes. Note any config or ABI changes.

## Coordinate With CLAUDE.md
- Treat `CLAUDE.md` as a companion playbook. I align terminology and reuse its templates (experiments, logging). When plans change materially, mirror the update into `CLAUDE.md` to keep both agents in sync.


## Change Policy For This Agent
- Do not modify existing code unless explicitly requested by the user.
- When proposing fixes or improvements, draft detailed plans and minimal diffs in plain `.txt` documents (path- and function-specific), for review before any code changes.
- Prefer small, traceable changes with clear acceptance criteria, diagnostics, and rollback notes.

## Handoff To Claude Code
- Audience-first docs: Proposal `.txt` files are written for a coding agent (Claude Code) to execute without extra context.
- Structure every proposal with:
  - Summary: one-paragraph goal, scope, and constraints.
  - Rationale: why changes are needed; link to symptoms and metrics.
  - Minimal Diffs: file paths and function names with precise, copy-pastable patches; avoid ambiguous instructions.
  - Implementation Steps: numbered, small-grain steps Claude can apply sequentially.
  - Config/Flags: new keys or toggles with defaults and compatibility notes.
  - Validation: exact commands, expected logs/metrics, acceptance criteria, and failure triage.
  - Rollback: how to revert and risk notes.
- Conventions:
  - Put proposal files near the affected code (same folder) with descriptive names, e.g., `*_patch_proposals.txt` and `*_diagnosis.txt`.
  - Use clear markers for code blocks and reference import paths (e.g., `entropy_experiments/probe_components.py:_teacher_force_logprobs`).
  - Prefer changes gated behind config flags; keep defaults backward compatible.
  - Note potential large-object allocations and peak memory expectations when relevant.
