# Repository Guidelines

## Project Structure & Module Organization
- `entropy_experiments/` hosts offline probes (for example `delta_entropy_approx.py` and `entropy_experiment_runner.py`) with configs in `configs/` and outputs in `results/`; copy `config_template.yaml` before customizing runs.
- `rl_training/` covers online training; keep run configs in `cfg/`, algorithm pieces in `algs/` and `schedulers/`, orchestration in `runners/`, and long-running artifacts in `results/`.
- Documentation and planning artifacts live in `docs/`, `CLAUDE.md`, and the companion guides. Store automation helpers in `scripts/`, and persist checkpoints or adapters in `checkpoints/` and `models/`.

## Build, Test, and Development Commands
- `python -m venv .venv && .venv\Scripts\activate` creates the Windows virtual environment.
- `pip install -e .[dev]` installs runtime dependencies plus linting and test tooling.
- `ruff check .` and `ruff format .` keep style consistent; run them before opening a PR.
- `pytest -q` runs the suite; narrow to `pytest -q tests/algs` or similar while iterating.
- Offline studies typically start with `python run_entropy_experiments.py --config entropy_experiments/configs/config_template.yaml --dry-run` before launching a full sweep.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation, snake_case functions and modules, PascalCase classes, and UPPER_CASE constants.
- Type hint public entry points and add concise Google-style docstrings that describe tensor shapes or rollout assumptions.
- Order imports stdlib -> third-party -> local, avoid wildcard imports, and stay ASCII unless extending a file that already uses symbols.

## Testing Guidelines
- Use `pytest`; mirror module paths as `tests/<area>/test_<topic>.py` and seed randomness (`torch.manual_seed`, `numpy.random.seed`) for deterministic metrics.
- Maintain >=85% statement coverage across `rl_training/*` and critical experiment utilities, storing fixtures locally instead of fetching from remote storage.

## Commit & Pull Request Guidelines
- Commit messages follow `scope: imperative summary`, e.g., `trainer: add entropy probe hook`; group related changes and avoid mixing refactors with features.
- PRs must state purpose, headline diffs, config or flag changes, validation commands, and links to logs or figures, plus rollback notes when training loops shift.

## Agent Coordination Notes
- Draft `*_patch_proposals.txt` beside target modules before editing, following the Summary -> Rationale -> Minimal Diffs -> Steps -> Validation -> Rollback template in `CLAUDE.md`.
- Keep CLAUDE and AGENTS guides synchronized whenever workflows or expectations change so human contributors and paired agents stay aligned.
